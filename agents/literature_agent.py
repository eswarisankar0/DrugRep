import os
import time
import json
from dotenv import load_dotenv
from Bio import Entrez
load_dotenv()

ENTREZ_EMAIL = os.getenv("NCBI_EMAIL")
ENTREZ_API_KEY = os.getenv("NCBI_API_KEY")                         

DISEASE_NAME = "Parkinson disease"           # PubMed search term for the disease
MAX_ABSTRACTS_PER_PAIR = 15

OUTPUT_DIR = "output/raw_literature"

# Manual test list — swap this out for reading combined_candidates.csv once fetch is verified
TEST_DRUGS = ["Amantadine", "Ketamine"]


def configure_entrez():
    Entrez.email = ENTREZ_EMAIL
    if ENTREZ_API_KEY:
        Entrez.api_key = ENTREZ_API_KEY


def search_pubmed(drug_name: str, disease_name: str, max_results: int) -> list[str]:
    """Return a list of PMIDs matching the drug + disease query."""
    query = f"({drug_name}[Title/Abstract]) AND ({disease_name}[Title/Abstract])"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Given PMIDs, fetch title, journal, year, abstract text for each."""
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

            results.append({
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "abstract": abstract,
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

    for drug in TEST_DRUGS:
        result = fetch_for_pair(drug, DISEASE_NAME)

        safe_drug = drug.lower().replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_drug}_parkinsons.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"  Saved to {out_path}\n")
        time.sleep(0.4)


if __name__ == "__main__":
    main()