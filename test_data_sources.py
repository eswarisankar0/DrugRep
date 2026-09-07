import os
import requests


TIMEOUT = 20


def test_open_targets():
    print("\n[1/7] Open Targets")

    url = "https://api.platform.opentargets.org/api/v4/graphql"

    query = """
    query {
        target(ensemblId: "ENSG00000141510") {
            id
            approvedSymbol
            biotype
        }
    }
    """

    response = requests.post(
        url,
        json={"query": query},
        timeout=TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    target = data["data"]["target"]

    print("  Status:", response.status_code)
    print("  Target:", target["approvedSymbol"])
    print("  ID:", target["id"])

    return target


def test_pubmed():
    print("\n[2/7] PubMed / NCBI")

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": "drug repurposing",
        "retmode": "json",
        "retmax": 1,
        "tool": "drugrep_test",
        "email": os.getenv("NCBI_EMAIL"),
    }

    api_key = os.getenv("NCBI_API_KEY")

    if api_key:
        params["api_key"] = api_key

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    ids = data["esearchresult"]["idlist"]

    print("  Status:", response.status_code)
    print("  PMID:", ids[0] if ids else "No result")

    return ids


def test_clinical_trials():
    print("\n[3/7] ClinicalTrials.gov")

    url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.term": "Parkinson disease",
        "pageSize": 1,
    }

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    studies = data.get("studies", [])

    print("  Status:", response.status_code)

    if studies:
        study = studies[0]
        protocol = study["protocolSection"]

        print(
            "  Trial:",
            protocol["identificationModule"]["nctId"]
        )
        print(
            "  Title:",
            protocol["identificationModule"]["briefTitle"]
        )

    return studies


def test_chembl():
    print("\n[4/7] ChEMBL")

    url = "https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json"

    response = requests.get(
        url,
        timeout=TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    print("  Status:", response.status_code)
    print("  ChEMBL ID:", data.get("molecule_chembl_id"))
    print("  Name:", data.get("pref_name"))

    return data


def test_reactome():
    print("\n[5/7] Reactome")

    url = "https://reactome.org/ContentService/data/database/version"

    response = requests.get(
        url,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    print("  Status:", response.status_code)
    print("  Reactome version:", response.text.strip())

    return response.text.strip()


def test_string():
    print("\n[6/7] STRING")

    url = "https://string-db.org/api/json/get_string_ids"

    params = {
        "identifiers": "TP53",
        "species": 9606,
        "limit": 1,
        "caller_identity": "drugrep_test",
    }

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    print("  Status:", response.status_code)

    if data:
        print("  Input:", data[0].get("queryItem"))
        print("  STRING ID:", data[0].get("stringId"))

    return data


def test_openfda():
    print("\n[7/7] OpenFDA / FAERS")

    url = "https://api.fda.gov/drug/event.json"

    params = {
        "limit": 1
    }

    api_key = os.getenv("OPENFDA_API_KEY")

    if api_key:
        params["api_key"] = api_key

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])

    print("  Status:", response.status_code)
    print("  Records returned:", len(results))

    if results:
        print(
            "  Report ID:",
            results[0].get("safetyreportid")
        )

    return results


def main():

    tests = [
        ("Open Targets", test_open_targets),
        ("PubMed", test_pubmed),
        ("ClinicalTrials.gov", test_clinical_trials),
        ("ChEMBL", test_chembl),
        ("Reactome", test_reactome),
        ("STRING", test_string),
        ("OpenFDA", test_openfda),
    ]

    passed = 0

    print("=" * 60)
    print("DRUGREP — EXTERNAL DATA SOURCE CONNECTIVITY TEST")
    print("=" * 60)

    for name, test_function in tests:

        try:
            test_function()
            print(f"  ✅ {name} PASSED")
            passed += 1

        except Exception as e:
            print(f"  ❌ {name} FAILED")
            print(f"     {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{len(tests)} sources passed")
    print("=" * 60)


if __name__ == "__main__":
    main()