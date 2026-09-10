from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class EvidenceRecord:
    drug_id: str                     # ChEMBL / DrugBank id if available, else drug_name
    drug_name: str
    disease_id: str                  # e.g. DOID:14330 or EFO id
    disease_name: str

    evidence_type: str                # e.g. "rct", "case_report", "preclinical",
                                       # "network_prediction", "target_association"
    source: str                       # e.g. "pubmed", "open_targets", "hetionet"
    source_id: str                    # PMID / NCT id / ChEMBL id / etc.

    finding_summary: str               # one or two sentence plain-language summary
    confidence_score: float            # 0-1, meaning defined per evidence_type

    date: Optional[str] = None         # publication date or data date, ISO format if possible
    raw_data_ref: Optional[str] = None  # URL or file path back to the raw source

    agent_name: str = "unassigned"     # which agent produced this record

    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"confidence_score must be between 0 and 1, got {self.confidence_score}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def to_json_file(records: list["EvidenceRecord"], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)

    @staticmethod
    def from_json_file(path: str) -> list["EvidenceRecord"]:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return [EvidenceRecord(**row) for row in rows]


if __name__ == "__main__":
    # quick smoke test
    r = EvidenceRecord(
        drug_id="CHEMBL1569",
        drug_name="Amantadine",
        disease_id="DOID:14330",
        disease_name="Parkinson's disease",
        evidence_type="rct",
        source="pubmed",
        source_id="PMID:12345678",
        finding_summary="Example finding for schema validation.",
        confidence_score=0.7,
        date="2020-01-01",
        raw_data_ref="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        agent_name="literature_agent",
    )
    print(r.to_dict())