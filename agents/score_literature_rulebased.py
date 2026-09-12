"""
agents/score_literature_rulebased.py

A fully rule-based (no LLM, no API key, no cost) alternative to
score_literature.py's real_score_abstract(). Instead of asking an LLM to
classify evidence type, this reads PubMed's own <PublicationType> tags,
which already tell you the study design directly from NCBI's own
curation. The LLM's real value-add was the finding_summary and nuanced
confidence — this version replaces those with a fixed evidence-type
hierarchy and a simple heuristic confidence score.

Trade-off vs LLM scoring:
  + Zero cost, zero rate limits, deterministic, instant
  + No API key needed at all — safe to run for every candidate, every time
  - No natural-language finding_summary (just a templated sentence)
  - Confidence score is a coarse heuristic, not a nuanced read of the
    actual abstract content (e.g. can't tell a positive RCT from a
    negative one — just that it IS an RCT)

Good fit as: a free first-pass filter, or the default scoring mode for a
public-facing deployment, while a smaller/cheaper LLM pass (or your own
Anthropic budget) is reserved only for the shortlist of candidates that
actually make it to top rankings.

Requires PublicationType data, which comes from literature_agent.py's
raw JSON IF you extend fetch_abstracts() there to also capture
article_data["PublicationTypeList"]. See NOTE at bottom.
"""

import json
import glob
import os

from schema import EvidenceRecord

RAW_LITERATURE_DIR = "output/raw_literature"
OUTPUT_PATH = "output/literature_evidence_rulebased.json"

DRUG_ID_LOOKUP = {
    "amantadine": "CHEMBL1569",
    "ketamine": "CHEMBL1714",
}
DISEASE_ID = "DOID:14330"

# Ranked strongest -> weakest. First match in an article's PublicationType
# list wins. Anything not in this map falls through to "unclassified".
EVIDENCE_TYPE_RULES = [
    ("Randomized Controlled Trial", "rct", 0.85),
    ("Clinical Trial, Phase III",   "rct", 0.80),
    ("Clinical Trial, Phase II",    "rct", 0.70),
    ("Clinical Trial",              "observational_study", 0.60),
    ("Observational Study",         "observational_study", 0.55),
    ("Comparative Study",           "observational_study", 0.50),
    ("Case Reports",                "case_report", 0.30),
    ("Review",                      "review", 0.20),
    ("Systematic Review",           "review", 0.35),
    ("Meta-Analysis",               "review", 0.45),
]

DEFAULT_TYPE = "preclinical"   # sensible fallback: most non-clinical PubMed
DEFAULT_CONFIDENCE = 0.25      # tags on drug-disease abstracts are lab/animal studies


def classify_publication_types(pub_types: list[str]) -> tuple[str, float]:
    """Walk the ranked rule list and return the strongest matching
    (evidence_type, base_confidence) pair found in this article's tags."""
    for tag, evidence_type, confidence in EVIDENCE_TYPE_RULES:
        if tag in pub_types:
            return evidence_type, confidence
    return DEFAULT_TYPE, DEFAULT_CONFIDENCE


def adjust_confidence_for_relevance(base_confidence: float, drug_name: str,
                                      disease_name: str, title: str, abstract: str) -> float:
    """Small heuristic bump/penalty based on keyword presence — a crude
    stand-in for the LLM's relevance judgment. Not a substitute for real
    semantic understanding, but catches the obvious cases."""
    text = f"{title} {abstract}".lower()
    score = base_confidence

    if drug_name.lower() not in text:
        score *= 0.3  # drug barely mentioned in the actual text
    if disease_name.lower().split()[0] not in text:  # e.g. "parkinson"
        score *= 0.3

    return round(min(score, 1.0), 2)


def rule_based_score_abstract(drug_name: str, disease_name: str, abstract_record: dict) -> dict:
    pub_types = abstract_record.get("publication_types", [])
    evidence_type, base_confidence = classify_publication_types(pub_types)

    confidence = adjust_confidence_for_relevance(
        base_confidence, drug_name, disease_name,
        abstract_record.get("title", ""), abstract_record.get("abstract", "")
    )

    finding_summary = (
        f"PubMed-tagged as {', '.join(pub_types) if pub_types else 'untyped'} "
        f"study mentioning {drug_name} and {disease_name}. "
        f"(Rule-based classification — read the abstract directly for actual findings.)"
    )

    return {
        "evidence_type": evidence_type,
        "confidence_score": confidence,
        "finding_summary": finding_summary,
    }


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
            continue

        scored = rule_based_score_abstract(drug_name, disease_name, abstract_record)

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
            agent_name="literature_agent_rulebased",
        )
        records.append(record)

    return records


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_records = []

    for filepath in glob.glob(os.path.join(RAW_LITERATURE_DIR, "*.json")):
        print(f"Processing {filepath}")
        records = process_file(filepath)
        print(f"  -> {len(records)} evidence records produced")
        all_records.extend(records)

    EvidenceRecord.to_json_file(all_records, OUTPUT_PATH)
    print(f"\nSaved {len(all_records)} total evidence records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# NOTE: to make this fully functional, literature_agent.py's
# fetch_abstracts() needs one addition — capture PublicationType tags:
#
#   pub_type_list = article_data.get("PublicationTypeList", [])
#   publication_types = [str(pt) for pt in pub_type_list]
#
# ...and add "publication_types": publication_types to the dict appended
# to `results`. This is free — it's already in the same XML response
# efetch returns, just not currently being read.
# ---------------------------------------------------------------------