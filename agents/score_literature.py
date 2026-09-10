"""
agents/score_literature.py

Reads raw abstracts fetched by literature_agent.py and scores each one,
producing structured EvidenceRecord entries (schema.py).

TWO MODES, controlled by USE_MOCK below:

  USE_MOCK = True   -> no API key needed. Uses a fake scorer so you can verify
                       the file I/O, schema validation, and error handling all
                       work correctly. Scores are NOT real evidence judgments.

  USE_MOCK = False  -> real scoring via the Anthropic API.
                       Requires: pip install anthropic
                       Requires: $env:ANTHROPIC_API_KEY="sk-ant-..." set in your terminal

Switch USE_MOCK to False the moment you have a real API key — nothing else
in this file needs to change.
"""

import os
import json
import time
import glob
import random

from schema import EvidenceRecord

# ---- TOGGLE THIS ----
USE_MOCK = True

MODEL = "claude-sonnet-4-6"

RAW_LITERATURE_DIR = "output/raw_literature"
OUTPUT_PATH = "output/literature_evidence.json"

DRUG_ID_LOOKUP = {
    "amantadine": "CHEMBL1569",
    "ketamine": "CHEMBL1714",
}
DISEASE_ID = "DOID:14330"  # Parkinson's disease


SCORING_PROMPT_TEMPLATE = """You are assisting a drug repurposing research pipeline. \
You will be given one PubMed abstract and a drug-disease pair. Assess how strongly \
this abstract supports the drug as a treatment for the disease.

Drug: {drug_name}
Disease: {disease_name}

Abstract title: {title}
Abstract text: {abstract}

Respond with ONLY a JSON object (no markdown fences, no preamble, no extra text) \
with exactly these fields:

{{
  "evidence_type": one of ["rct", "observational_study", "case_report", "preclinical", \
"review", "not_relevant"],
  "confidence_score": a number from 0.0 to 1.0, where 0.0 means the abstract provides \
no real support for this drug treating this disease, and 1.0 means strong, direct, \
well-controlled clinical evidence that it does,
  "finding_summary": one plain-language sentence (under 30 words) summarizing what this \
abstract actually found regarding this drug and this disease
}}

If the abstract is not actually about this drug-disease relationship, use \
evidence_type "not_relevant" and confidence_score 0.0.
"""


def mock_score_abstract(drug_name: str, disease_name: str, abstract_record: dict) -> dict:
    """Fake scorer — deterministic-ish, just enough variety to test the pipeline."""
    evidence_types = ["rct", "observational_study", "case_report", "preclinical", "review"]
    chosen_type = random.choice(evidence_types)
    return {
        "evidence_type": chosen_type,
        "confidence_score": round(random.uniform(0.1, 0.9), 2),
        "finding_summary": (
            f"[MOCK] Placeholder summary for {drug_name} and {disease_name} "
            f"based on PMID {abstract_record.get('pmid', 'unknown')}."
        ),
    }


def real_score_abstract(drug_name: str, disease_name: str, abstract_record: dict) -> dict:
    from anthropic import Anthropic  # imported here so USE_MOCK=True never needs this installed

    client = Anthropic()

    prompt = SCORING_PROMPT_TEMPLATE.format(
        drug_name=drug_name,
        disease_name=disease_name,
        title=abstract_record.get("title", ""),
        abstract=abstract_record.get("abstract", "")[:4000],
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    return json.loads(raw_text)


def score_abstract(drug_name: str, disease_name: str, abstract_record: dict) -> dict:
    if USE_MOCK:
        return mock_score_abstract(drug_name, disease_name, abstract_record)
    return real_score_abstract(drug_name, disease_name, abstract_record)


def process_file(filepath: str) -> list[EvidenceRecord]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    drug_name = data["drug_name"]
    disease_name = data["disease_name"]
    drug_id = DRUG_ID_LOOKUP.get(drug_name.lower(), drug_name)

    records = []

    for abstract_record in data.get("abstracts", []):
        pmid = abstract_record.get("pmid", "unknown")

        if not abstract_record.get("abstract"):
            print(f"  Skipping PMID {pmid} — no abstract text")
            continue

        print(f"  Scoring PMID {pmid}...")

        try:
            scored = score_abstract(drug_name, disease_name, abstract_record)

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
                agent_name="literature_agent" + ("_mock" if USE_MOCK else ""),
            )
            records.append(record)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  FAILED to score PMID {pmid}: {type(e).__name__}: {e}")
            continue

        if not USE_MOCK:
            time.sleep(0.3)  # only need pacing for real API calls

    return records


def main():
    mode_label = "MOCK (no API key used)" if USE_MOCK else "REAL (Anthropic API)"
    print(f"Running in {mode_label} mode.\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_records: list[EvidenceRecord] = []

    input_files = glob.glob(os.path.join(RAW_LITERATURE_DIR, "*.json"))
    print(f"Found {len(input_files)} raw literature file(s) to score.\n")

    for filepath in input_files:
        print(f"Processing {filepath}")
        records = process_file(filepath)
        print(f"  -> {len(records)} evidence records produced\n")
        all_records.extend(records)

    EvidenceRecord.to_json_file(all_records, OUTPUT_PATH)
    print(f"Saved {len(all_records)} total evidence records to {OUTPUT_PATH}")

    if USE_MOCK:
        print(
            "\nNOTE: these are MOCK scores for pipeline testing only — "
            "not real evidence assessments. Set USE_MOCK = False once you "
            "have an Anthropic API key to get real scores."
        )


if __name__ == "__main__":
    main()