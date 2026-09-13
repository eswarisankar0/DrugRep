"""
agents/verify_scoring.py

End-to-end check on the literature scoring stage. Run it after
score_literature.py to confirm the output is actually usable, rather than
inferring that from the absence of a traceback.

The failure this exists to catch: the hybrid scorer is designed to degrade
gracefully — a failed LLM call falls back to a templated summary, and cache
hits skip the API entirely. That makes a broken API path look exactly like
a cheap, successful run. Nothing crashes, nothing is logged, and the
records that come out have no findings in them. Only an assertion on the
output distinguishes the two.

Usage (from the agents/ directory):
    python verify_scoring.py

Exit code 0 = all checks passed, 1 = at least one FAIL.
"""

import glob
import json
import os
import sys
from collections import Counter

from schema import EvidenceRecord
from score_literature import (
    CACHE_PATH,
    DISEASE_DISPLAY_NAME,
    OUTPUT_PATH,
    RAW_LITERATURE_DIR,
    classify_publication_types,
    should_call_llm,
)

TEMPLATE_MARKER = "not sent for LLM summarization"

failures = 0
warnings = 0


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False):
    global failures, warnings
    if ok:
        status = "PASS"
    elif warn_only:
        status = "WARN"
        warnings += 1
    else:
        status = "FAIL"
        failures += 1
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))


def main():
    print("=" * 72)
    print("END-TO-END SCORING VERIFICATION")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Inputs exist
    # ---------------------------------------------------------------
    print("\n[1] Inputs")
    raw_files = glob.glob(os.path.join(RAW_LITERATURE_DIR, "*.json"))
    check("raw literature files present", bool(raw_files), f"{len(raw_files)} files")
    if not raw_files:
        print("\nNothing to verify — run literature_agent.py first.")
        return 1

    expected_abstracts = 0
    raw_drugs = set()
    for fp in raw_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_drugs.add(data["drug_name"])
        expected_abstracts += sum(1 for a in data.get("abstracts", []) if a.get("abstract"))
    print(f"        {len(raw_drugs)} drugs, {expected_abstracts} abstracts with text")

    # ---------------------------------------------------------------
    # 2. Output exists and parses through the schema
    # ---------------------------------------------------------------
    print("\n[2] Output file")
    check("output file exists", os.path.exists(OUTPUT_PATH), OUTPUT_PATH)
    if not os.path.exists(OUTPUT_PATH):
        print("\nNothing to verify — run score_literature.py first.")
        return 1

    # from_json_file() re-runs EvidenceRecord.__post_init__, so this also
    # revalidates every confidence_score is in [0, 1].
    records = EvidenceRecord.from_json_file(OUTPUT_PATH)
    check("records parse through EvidenceRecord schema", True, f"{len(records)} records")

    # ---------------------------------------------------------------
    # 3. Coverage — every drug and abstract accounted for
    # ---------------------------------------------------------------
    print("\n[3] Coverage")
    scored_drugs = {r.drug_name for r in records}
    missing = raw_drugs - scored_drugs
    check("every drug scored", not missing,
          f"{len(scored_drugs)}/{len(raw_drugs)}" +
          (f" — missing: {sorted(missing)[:5]}" if missing else ""))
    check("every abstract produced a record", len(records) == expected_abstracts,
          f"{len(records)} records vs {expected_abstracts} abstracts")

    # ---------------------------------------------------------------
    # 4. The blank-summary regression
    # ---------------------------------------------------------------
    print("\n[4] Summary quality (the regression this file exists for)")
    blank = [r for r in records if not r.finding_summary.strip()]
    check("no blank finding_summary", not blank, f"{len(blank)} blank")

    templated = [r for r in records if TEMPLATE_MARKER in r.finding_summary]
    real = [r for r in records if r.finding_summary.strip()
            and TEMPLATE_MARKER not in r.finding_summary]
    print(f"        {len(real)} LLM-written, {len(templated)} templated (by design)")

    # Every abstract the routing logic marks LLM-eligible should have come
    # back with a real summary. If these are templated instead, the API
    # path silently fell back on every call.
    eligible = 0
    for fp in raw_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ab in data.get("abstracts", []):
            if not ab.get("abstract"):
                continue
            evidence_type, _ = classify_publication_types(ab.get("publication_types", []))
            if should_call_llm(evidence_type, data["drug_name"]):
                eligible += 1
    check("LLM-eligible abstracts got real summaries", len(real) >= eligible * 0.95,
          f"{len(real)} real vs {eligible} eligible")
    check("real summaries are not truncated mid-word",
          all(r.finding_summary.rstrip().endswith((".", "!", "?")) for r in real),
          f"{sum(1 for r in real if not r.finding_summary.rstrip().endswith(('.', '!', '?')))} "
          f"lack terminal punctuation", warn_only=True)

    # ---------------------------------------------------------------
    # 5. Cache health
    # ---------------------------------------------------------------
    print("\n[5] Cache")
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        poisoned = [k for k, v in cache.items() if not v.strip()]
        check("no blank cache entries", not poisoned,
              f"{len(poisoned)} poisoned of {len(cache)}")
        print(f"        {len(cache)} PMIDs cached — a re-run reuses these at zero cost")
    else:
        check("cache file exists", False, "no cache written — were any LLM calls made?",
              warn_only=True)

    # ---------------------------------------------------------------
    # 6. Field hygiene
    # ---------------------------------------------------------------
    print("\n[6] Field hygiene")
    check("disease_name is the display form", all(r.disease_name == DISEASE_DISPLAY_NAME for r in records),
          f"found: {sorted({r.disease_name for r in records})}")
    check("all source_id are PMIDs", all(r.source_id.startswith("PMID:") for r in records))
    check("all raw_data_ref present", all(r.raw_data_ref for r in records))
    check("all records have a publication year",
          all(r.date for r in records),
          f"{sum(1 for r in records if not r.date)} missing", warn_only=True)

    # ---------------------------------------------------------------
    # 7. Distributions — sanity, not correctness
    # ---------------------------------------------------------------
    print("\n[7] Distributions")
    by_type = Counter(r.evidence_type for r in records)
    for evidence_type, n in by_type.most_common():
        print(f"        {evidence_type:22s} {n:4d}")
    check("more than one evidence_type present", len(by_type) > 1,
          f"{len(by_type)} distinct types")

    confidences = [r.confidence_score for r in records]
    print(f"        confidence: min {min(confidences):.2f}, "
          f"max {max(confidences):.2f}, mean {sum(confidences)/len(confidences):.2f}")

    # ---------------------------------------------------------------
    # 8. Eyeball the strongest evidence
    # ---------------------------------------------------------------
    print("\n[8] Top 5 records by confidence (read these — do they make sense?)")
    for r in sorted(records, key=lambda r: r.confidence_score, reverse=True)[:5]:
        print(f"        {r.confidence_score:.2f}  {r.drug_name} [{r.evidence_type}] {r.source_id}")
        print(f"              {r.finding_summary[:110]}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    if failures:
        print(f"RESULT: {failures} check(s) FAILED, {warnings} warning(s)")
    else:
        print(f"RESULT: all checks passed ({warnings} warning(s))")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
