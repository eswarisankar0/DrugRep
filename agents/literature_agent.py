import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from Bio import Entrez
load_dotenv()

ENTREZ_EMAIL = os.getenv("NCBI_EMAIL")
ENTREZ_API_KEY = os.getenv("NCBI_API_KEY")

DISEASE_NAME = "Parkinson"                   # base name only — search_pubmed() builds
                                              # all the "Parkinson disease" / "Parkinson's
                                              # disease" / MeSH variants from this
MAX_ABSTRACTS_PER_PAIR = 15

OUTPUT_DIR = "output/raw_literature"

# Real candidates, read from Day 1's output instead of a manual test list.
# Path is relative to wherever you run this script from (agents/ by default).
COMBINED_CANDIDATES_PATH = "../data_ingestion/combined_candidates.csv"

#NCBI rate limits mean ~77 candidates will take a few minutes even with an API key.
LIMIT_CANDIDATES = None


def load_candidate_drug_names(csv_path: str, limit: int = None) -> list[str]:
    """Read combined_candidates.csv and return a deduplicated list of
    clean drug names to search PubMed with.

    Uses normalized_name (salt-suffix-stripped, e.g. 'amantadine' instead
    of 'AMANTADINE HYDROCHLORIDE') so we don't fire off two near-identical
    PubMed searches for what is really the same compound. Rows are already
    sorted by n_sources then sort_score from combine_candidates.py, so
    dual-source and top-ranked candidates naturally come first — useful
    when LIMIT_CANDIDATES is set during testing.
    """
    df = pd.read_csv(csv_path)

    seen = set()
    ordered_names = []
    for name in df["normalized_name"]:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)

    if limit is not None:
        ordered_names = ordered_names[:limit]

    # Title-case for a cleaner PubMed query / output filename
    # (normalized_name is lowercase, e.g. "amantadine")
    return [name.title() for name in ordered_names]


def configure_entrez():
    Entrez.email = ENTREZ_EMAIL
    if ENTREZ_API_KEY:
        Entrez.api_key = ENTREZ_API_KEY


def search_pubmed(drug_name: str, disease_name: str, max_results: int) -> list[str]:
    """Return a list of PMIDs matching the drug + disease query.

    IMPORTANT: a bare phrase like "Parkinson disease[Title/Abstract]" only
    matches that EXACT phrase in PubMed's phrase index. Most modern papers
    write "Parkinson's disease" (with apostrophe) instead, so a plain,
    unquoted, single-form disease phrase silently misses a large fraction
    of real, relevant literature. Fixing this by combining:
      - the official MeSH heading (catches indexed papers regardless of
        exact wording used in the text)
      - both the possessive and non-possessive text forms, quoted
    combined with OR, so any of the three forms is enough to match.
    """
    disease_clause = (
        f'("{disease_name} disease"[MeSH Terms] '
        f'OR "{disease_name}\'s disease"[Title/Abstract] '
        f'OR "{disease_name} disease"[Title/Abstract])'
    )
    query = f"({drug_name}[Title/Abstract]) AND {disease_clause}"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Given PMIDs, fetch title, journal, year, abstract text, and
    publication type(s) for each."""
    if not pmids:
        return []

    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    results = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            article_data = medline["Article"]

            title = str(article_data.get("ArticleTitle", ""))

            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(part) for part in abstract_parts)

            journal = str(article_data.get("Journal", {}).get("Title", ""))

            year = None
            pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            if "Year" in pub_date:
                year = str(pub_date["Year"])

            pub_type_list = article_data.get("PublicationTypeList", [])
            publication_types = [str(pt) for pt in pub_type_list]

            results.append({
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "publication_types": publication_types,
                "raw_data_ref": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        except (KeyError, IndexError):
            # some records have unusual structure (e.g. no abstract) — skip rather than crash
            continue

    return results


def fetch_for_pair(drug_name: str, disease_name: str) -> dict:
    print(f"Searching PubMed for: {drug_name} + {disease_name}")
    pmids = search_pubmed(drug_name, disease_name, MAX_ABSTRACTS_PER_PAIR)
    print(f"  Found {len(pmids)} PMIDs")

    time.sleep(0.4)  # be polite to NCBI's rate limit even with an API key

    abstracts = fetch_abstracts(pmids)
    print(f"  Retrieved {len(abstracts)} abstracts with usable text")

    return {
        "drug_name": drug_name,
        "disease_name": disease_name,
        "n_abstracts": len(abstracts),
        "abstracts": abstracts,
    }


def main():
    configure_entrez()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    drug_names = load_candidate_drug_names(COMBINED_CANDIDATES_PATH, limit=LIMIT_CANDIDATES)
    print(f"Loaded {len(drug_names)} candidate drug(s) from {COMBINED_CANDIDATES_PATH}")
    if LIMIT_CANDIDATES is not None:
        print(f"(LIMIT_CANDIDATES={LIMIT_CANDIDATES} — set to None in the script to run all candidates)\n")
    else:
        print()

    for drug in drug_names:
        result = fetch_for_pair(drug, DISEASE_NAME)

        safe_drug = drug.lower().replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_drug}_parkinsons.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"  Saved to {out_path}\n")
        time.sleep(0.4)


if __name__ == "__main__":
    main()