"""
agents/score_literature_hybrid.py  (FINAL — priority-tier wired in)

Cost-optimized hybrid scorer, three layers of cost control:

  1. EVERY abstract gets a free, instant evidence_type + confidence_score
     from PubMed's own PublicationType tags (classify_publication_types,
     imported from score_literature_rulebased.py).

  2. The LLM is only called to generate finding_summary, and only when:
       - evidence_type is clinically strong (rct / observational_study /
         case_report) for ANY candidate, OR
       - the drug is in the priority set (dual-source hits, or top-N by
         score within each source — derived automatically from
         combined_candidates.csv via priority_candidates.py)
     Everything else gets a free templated summary instead.

  3. A local on-disk cache keyed by PMID means the same paper is NEVER
     summarized twice, even across multiple drugs or repeated runs.

Requires: pip install anthropic pandas
Requires: ANTHROPIC_API_KEY set in your terminal session before running,
          e.g. (PowerShell) $env:ANTHROPIC_API_KEY="sk-ant-..."
"""

import os
import json
import time
import glob

from schema import EvidenceRecord
from score_literature_rulebased import (
    classify_publication_types,
    adjust_confidence_for_relevance,
)
from priority_candidates import get_priority_candidates
from dotenv import load_dotenv

# override=True makes agents/.env the single source of truth for the key.
# The default (override=False) lets an exported shell variable silently
# shadow this file, so rotating the key in .env appears to do nothing and
# the run keeps failing on the stale value.
load_dotenv(override=True)

# Haiku 4.5 is Anthropic's current cheapest model ($1/$5 per million input/
# output tokens) — plenty capable for a short one-sentence summarization
# task like this. Sonnet 5 would work too but costs 2x for no real
# quality benefit on a task this simple.
MODEL = "claude-haiku-4-5"

# One sentence needs ~30 output tokens; 256 is headroom so a slightly
# chatty reply is never truncated mid-word. Output is billed per token
# actually generated, so a larger cap costs nothing when replies are short.
MAX_SUMMARY_TOKENS = 256

RAW_LITERATURE_DIR = "output/raw_literature"
OUTPUT_PATH = "output/literature_evidence.json"
CACHE_PATH = "output/finding_summary_cache.json"

DISEASE_ID = "DOID:14330"  # Parkinson's disease

# The raw literature JSONs store disease_name as "Parkinson" — that's the
# base string literature_agent.py uses to BUILD its PubMed query variants,
# not a display name. Use the proper name in prompts and in the evidence
# records; keep the raw one for the keyword relevance heuristic, which
# does disease_name.split()[0] and would stop matching on "Parkinson's".
DISEASE_DISPLAY_NAME = "Parkinson's disease"

# Drug-name -> ChEMBL/DrugBank id lookup for records we know; falls back
# to the plain drug name if not listed here. Extend as you go.
DRUG_ID_LOOKUP = {
    "amantadine": "CHEMBL1569",
    "ketamine": "CHEMBL1714",
}

# Evidence types worth an LLM's nuanced read REGARDLESS of which drug
# they're about — clinical evidence is high-value everywhere.
ALWAYS_LLM_TYPES = {"rct", "observational_study", "case_report"}

# Loaded once at import time from combined_candidates.csv — dual-source
# hits and top-N-per-source candidates. See priority_candidates.py.
try:
    PRIORITY_CANDIDATES = get_priority_candidates()
except FileNotFoundError:
    print("WARNING: combined_candidates.csv not found — priority tier "
          "disabled, only ALWAYS_LLM_TYPES will trigger real LLM calls.")
    PRIORITY_CANDIDATES = set()

SUMMARY_PROMPT_TEMPLATE = """You are assisting a drug repurposing research pipeline.
Given this PubMed abstract about {drug_name} and {disease_name}, write ONE plain-language
sentence (under 30 words) summarizing what this specific study actually found.

Title: {title}
Abstract: {abstract}

Respond with ONLY the sentence. No JSON, no quotes, no preamble."""


def normalize(name: str) -> str:
    return name.strip().lower()


def should_call_llm(evidence_type: str, drug_name: str) -> bool:
    """Decide whether this abstract earns a real LLM call, or gets the
    free templated summary instead."""
    if evidence_type in ALWAYS_LLM_TYPES:
        return True
    return normalize(drug_name) in PRIORITY_CANDIDATES


def load_cache() -> dict:
    """Load the PMID -> summary cache, dropping any blank entries.

    A blank summary is a failure that got memoized. Because the cache is
    consulted before the API and a hit short-circuits the call, a single
    bad run can permanently pin every affected PMID to an empty string —
    the pipeline then looks like it's working (cache hits, no errors, no
    cost) while producing records with no findings in them. Discarding
    blanks on load means a fixed run heals the cache instead of inheriting
    it, and costs only a re-summarization of the PMIDs that failed.
    """
    if not os.path.exists(CACHE_PATH):
        return {}

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    clean = {pmid: summary for pmid, summary in cache.items() if summary.strip()}
    dropped = len(cache) - len(clean)
    if dropped:
        print(f"Discarded {dropped} blank cached summary/summaries — these will be "
              f"re-summarized this run.")
    return clean


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_api_key() -> str:
    """Read and sanitize ANTHROPIC_API_KEY.

    The .strip() is not cosmetic. An API key with a stray trailing space
    or newline — trivially easy to introduce via `$env:ANTHROPIC_API_KEY=
    "sk-ant-... "` or a copy-paste into .env — is an illegal HTTP header
    value. httpx rejects it locally before any request goes out, and the
    SDK surfaces that as APIConnectionError("Connection error."), which
    looks exactly like a network outage and sends you hunting for a
    firewall problem that doesn't exist.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Get one at "
            "https://console.anthropic.com/settings/keys and set it as an "
            "environment variable (or in a .env file) before running this script."
        )
    return api_key


def extract_text(response) -> str:
    """Pull the text out of a response across all text blocks.

    Reading response.content[0].text assumes the first block is text.
    That's not guaranteed — a thinking block, or a response with no
    content at all, makes it either wrong or an exception. Concatenating
    every text block is correct regardless of what else is present.
    """
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def get_llm_finding_summary(drug_name: str, disease_name: str, abstract_record: dict,
                              max_retries: int = 3) -> str:
    # Imported here so a missing key/package never blocks the free
    # rule-based path.
    from anthropic import (
        Anthropic,
        APIConnectionError,
        APIStatusError,
        AuthenticationError,
        NotFoundError,
        RateLimitError,
    )

    client = Anthropic(api_key=get_api_key())
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        drug_name=drug_name,
        disease_name=disease_name,
        title=abstract_record.get("title", ""),
        abstract=abstract_record.get("abstract", "")[:4000],
    )

    # Transient failures shouldn't kill a long run that's already made many
    # real API calls — but a bad key or a bad model name will fail the same
    # way on every attempt, so those raise immediately instead of burning
    # three backoff sleeps to reach the same conclusion.
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_SUMMARY_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        except (AuthenticationError, NotFoundError):
            raise
        except APIConnectionError as e:
            last_error = e
            wait = 2 ** attempt
            print(f"    Network error on attempt {attempt}/{max_retries}, retrying in {wait}s...")
            time.sleep(wait)
            continue
        except RateLimitError as e:
            last_error = e
            wait = 30 * attempt
            print(f"    Rate limited on attempt {attempt}/{max_retries}, retrying in {wait}s...")
            time.sleep(wait)
            continue
        except APIStatusError as e:
            if e.status_code < 500:
                raise  # client error — retrying won't change the outcome
            last_error = e
            wait = 2 ** attempt
            print(f"    Server error {e.status_code} on attempt {attempt}/{max_retries}, "
                  f"retrying in {wait}s...")
            time.sleep(wait)
            continue

        if response.stop_reason == "refusal":
            print(f"    Model declined to summarize PMID "
                  f"{abstract_record.get('pmid', 'unknown')} — using templated summary.")
            return ""
        if response.stop_reason == "max_tokens":
            print(f"    Summary hit the {MAX_SUMMARY_TOKENS}-token cap and was truncated "
                  f"— raise MAX_SUMMARY_TOKENS.")

        return extract_text(response)

    raise last_error


def templated_summary(evidence_type: str, drug_name: str, disease_name: str, pub_types: list) -> str:
    label = ", ".join(pub_types) if pub_types else evidence_type
    return (
        f"{label} evidence mentioning {drug_name} and {disease_name} "
        f"(not sent for LLM summarization — see abstract for details)."
    )


def score_abstract_hybrid(drug_name: str, disease_name: str, abstract_record: dict, cache: dict) -> dict:
    pub_types = abstract_record.get("publication_types", [])
    evidence_type, base_confidence = classify_publication_types(pub_types)
    confidence = adjust_confidence_for_relevance(
        base_confidence, drug_name, disease_name,
        abstract_record.get("title", ""), abstract_record.get("abstract", "")
    )

    pmid = abstract_record.get("pmid", "unknown")

    if not should_call_llm(evidence_type, drug_name):
        finding_summary = templated_summary(
            evidence_type, drug_name, DISEASE_DISPLAY_NAME, pub_types)
        return {"evidence_type": evidence_type, "confidence_score": confidence,
                "finding_summary": finding_summary, "llm_called": False}

    if pmid in cache:
        finding_summary = cache[pmid]
        return {"evidence_type": evidence_type, "confidence_score": confidence,
                "finding_summary": finding_summary, "llm_called": False}

    finding_summary = get_llm_finding_summary(
        drug_name, DISEASE_DISPLAY_NAME, abstract_record
    )

    # Only memoize a real summary. Caching a blank would make this failure
    # permanent for this PMID — see load_cache().
    if not finding_summary:
        return {"evidence_type": evidence_type, "confidence_score": confidence,
                "finding_summary": templated_summary(
                    evidence_type, drug_name, DISEASE_DISPLAY_NAME, pub_types),
                "llm_called": True}

    cache[pmid] = finding_summary
    return {"evidence_type": evidence_type, "confidence_score": confidence,
            "finding_summary": finding_summary, "llm_called": True}


def process_file(filepath: str, cache: dict) -> tuple[list, int]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    drug_name = data["drug_name"]
    disease_name = data["disease_name"]
    drug_id = DRUG_ID_LOOKUP.get(drug_name.lower(), drug_name)

    records = []
    llm_calls_made = 0

    for abstract_record in data.get("abstracts", []):
        pmid = abstract_record.get("pmid", "unknown")
        if not abstract_record.get("abstract"):
            continue

        scored = score_abstract_hybrid(drug_name, disease_name, abstract_record, cache)
        if scored["llm_called"]:
            llm_calls_made += 1
            time.sleep(0.3)

        record = EvidenceRecord(
            drug_id=drug_id,
            drug_name=drug_name,
            disease_id=DISEASE_ID,
            disease_name=DISEASE_DISPLAY_NAME,
            evidence_type=scored["evidence_type"],
            source="pubmed",
            source_id=f"PMID:{pmid}",
            finding_summary=scored["finding_summary"],
            confidence_score=float(scored["confidence_score"]),
            date=abstract_record.get("year"),
            raw_data_ref=abstract_record.get("raw_data_ref"),
            agent_name="literature_agent_hybrid",
        )
        records.append(record)

    return records, llm_calls_made


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cache = load_cache()

    print(f"Priority candidates loaded: {len(PRIORITY_CANDIDATES)}")
    print(f"({', '.join(sorted(PRIORITY_CANDIDATES)) if PRIORITY_CANDIDATES else 'none'})\n")

    # Resume support: pick up where a previous (possibly crashed) run left
    # off, instead of starting from zero and losing already-completed work.
    all_records = []
    if os.path.exists(OUTPUT_PATH):
        try:
            all_records = EvidenceRecord.from_json_file(OUTPUT_PATH)
            print(f"Resuming: loaded {len(all_records)} existing evidence records from a previous run.")
        except (json.JSONDecodeError, FileNotFoundError):
            all_records = []

    # Discard every record belonging to a drug that produced any
    # summary-less record, so the resume gate below re-processes that drug
    # from scratch. Without this the gate would recognize the drug name,
    # skip its file, and keep the blank record forever — a previous failed
    # run would silently become permanent output.
    blank_drugs = {r.drug_name for r in all_records if not r.finding_summary.strip()}
    if blank_drugs:
        kept = [r for r in all_records if r.drug_name not in blank_drugs]
        print(f"Dropped {len(all_records) - len(kept)} record(s) across "
              f"{len(blank_drugs)} drug(s) with empty summaries — re-processing those.")
        all_records = kept

    already_processed_drugs = {r.drug_name for r in all_records}
    if already_processed_drugs:
        print(f"Skipping {len(already_processed_drugs)} drug(s) already completed in a previous run.\n")

    total_llm_calls = 0
    total_abstracts = 0

    filepaths = glob.glob(os.path.join(RAW_LITERATURE_DIR, "*.json"))

    try:
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                peek_drug_name = json.load(f)["drug_name"]

            if peek_drug_name in already_processed_drugs:
                continue

            print(f"Processing {filepath}")
            records, llm_calls = process_file(filepath, cache)
            total_llm_calls += llm_calls
            total_abstracts += len(records)
            print(f"  -> {len(records)} records, {llm_calls} required a real LLM call")
            all_records.extend(records)

            # Save after EVERY file so a crash never loses more than the
            # one file in progress.
            save_cache(cache)
            EvidenceRecord.to_json_file(all_records, OUTPUT_PATH)

    except KeyboardInterrupt:
        print("\nInterrupted by user — progress up to the last completed file is saved.")
    except Exception as e:
        print(f"\nStopped early due to an error: {type(e).__name__}: {e}")
        print("Progress up to the last completed file has already been saved.")
        print("Re-run this script to resume — cached PMIDs won't be re-charged.")
        raise
    finally:
        save_cache(cache)
        EvidenceRecord.to_json_file(all_records, OUTPUT_PATH)

    print(f"\nSaved {len(all_records)} total evidence records to {OUTPUT_PATH}")
    print(f"LLM calls made this run: {total_llm_calls} / {total_abstracts} abstracts "
          f"({100 * total_llm_calls / max(total_abstracts,1):.0f}%)")
    print(f"Cache now holds {len(cache)} previously-summarized PMIDs — "
          f"re-running this script will reuse them at zero cost.")


if __name__ == "__main__":
    main()