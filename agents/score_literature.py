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

MODEL = "claude-sonnet-4-6"

RAW_LITERATURE_DIR = "output/raw_literature"
OUTPUT_PATH = "output/literature_evidence.json"
CACHE_PATH = "output/finding_summary_cache.json"

DISEASE_ID = "DOID:14330"  # Parkinson's disease

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
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_llm_finding_summary(drug_name: str, disease_name: str, abstract_record: dict) -> str:
    from anthropic import Anthropic  # imported here so a missing key/package
                                      # never blocks the free rule-based path

    client = Anthropic()
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        drug_name=drug_name,
        disease_name=disease_name,
        title=abstract_record.get("title", ""),
        abstract=abstract_record.get("abstract", "")[:4000],
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


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
        finding_summary = templated_summary(evidence_type, drug_name, disease_name, pub_types)
        return {"evidence_type": evidence_type, "confidence_score": confidence,
                "finding_summary": finding_summary, "llm_called": False}

    if pmid in cache:
        finding_summary = cache[pmid]
        return {"evidence_type": evidence_type, "confidence_score": confidence,
                "finding_summary": finding_summary, "llm_called": False}

    finding_summary = get_llm_finding_summary(drug_name, disease_name, abstract_record)
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
            disease_name=disease_name,
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

    all_records = []
    total_llm_calls = 0
    total_abstracts = 0

    for filepath in glob.glob(os.path.join(RAW_LITERATURE_DIR, "*.json")):
        print(f"Processing {filepath}")
        records, llm_calls = process_file(filepath, cache)
        total_llm_calls += llm_calls
        total_abstracts += len(records)
        print(f"  -> {len(records)} records, {llm_calls} required a real LLM call")
        all_records.extend(records)

    save_cache(cache)
    EvidenceRecord.to_json_file(all_records, OUTPUT_PATH)

    print(f"\nSaved {len(all_records)} total evidence records to {OUTPUT_PATH}")
    print(f"LLM calls made this run: {total_llm_calls} / {total_abstracts} abstracts "
          f"({100 * total_llm_calls / max(total_abstracts,1):.0f}%)")
    print(f"Cache now holds {len(cache)} previously-summarized PMIDs — "
          f"re-running this script will reuse them at zero cost.")


if __name__ == "__main__":
    main()