import time
import csv
import requests


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

DISEASE_NAME = "Parkinson disease"

# Number of Parkinson-associated targets to retrieve
TOP_N_TARGETS = 15

# Maximum drugs to keep for each target
MAX_DRUGS_PER_TARGET = 10

# Output file
OUTPUT_CSV = "parkinsons_candidates.csv"


# =========================================================
# RUN GRAPHQL QUERY
# =========================================================

def run_query(query: str, variables: dict) -> dict:

    resp = requests.post(
        API_URL,
        json={
            "query": query,
            "variables": variables
        },
        timeout=30
    )

    print(f"HTTP status: {resp.status_code}")

    # If Open Targets returns an HTTP error,
    # print the actual response so we can diagnose it.
    if resp.status_code != 200:
        print("\n========== OPEN TARGETS ERROR ==========")
        print(resp.text)
        print("========================================\n")

    resp.raise_for_status()

    data = resp.json()

    # GraphQL can return HTTP 200 but still contain errors.
    if "errors" in data:

        print("\n========== GRAPHQL ERROR ==========")

        for error in data["errors"]:
            print(error)

        print("===================================\n")

        raise RuntimeError(
            "Open Targets GraphQL query failed."
        )

    return data["data"]


# =========================================================
# FIND PARKINSON DISEASE
# =========================================================

def find_disease_efo_id(name: str):

    query = """
    query Search($q: String!) {

        search(
            queryString: $q
            entityNames: ["disease"]
        ) {

            hits {
                id
                name
                entity
            }
        }
    }
    """

    data = run_query(
        query,
        {
            "q": name
        }
    )

    hits = data["search"]["hits"]

    if not hits:
        raise ValueError(
            f"No disease found for '{name}'"
        )

    # First search result
    disease_id = hits[0]["id"]
    disease_name = hits[0]["name"]

    return disease_id, disease_name


# =========================================================
# GET TOP PARKINSON-ASSOCIATED TARGETS
# =========================================================

def get_top_targets(disease_id: str, size: int):

    query = """
    query Targets(
        $diseaseId: String!
        $size: Int!
    ) {

        disease(
            efoId: $diseaseId
        ) {

            associatedTargets(
                page: {
                    index: 0
                    size: $size
                }
            ) {

                rows {

                    score

                    target {
                        id
                        approvedSymbol
                        approvedName
                    }
                }
            }
        }
    }
    """

    data = run_query(
        query,
        {
            "diseaseId": disease_id,
            "size": size
        }
    )

    disease = data["disease"]

    if not disease:
        raise ValueError(
            f"Could not retrieve disease: {disease_id}"
        )

    return disease["associatedTargets"]["rows"]


# =========================================================
# GET DRUG / CLINICAL CANDIDATES FOR A TARGET
# =========================================================

def get_known_drugs(
    ensembl_id: str,
    max_drugs: int
):

    query = """
    query DrugCandidates(
        $ensemblId: String!
    ) {

        target(
            ensemblId: $ensemblId
        ) {

            id
            approvedSymbol

            drugAndClinicalCandidates {

                count

                rows {

                    maxClinicalStage

                    drug {
                        id
                        name
                    }
                }
            }
        }
    }
    """

    data = run_query(
        query,
        {
            "ensemblId": ensembl_id
        }
    )

    target = data["target"]

    if not target:
        return []

    drug_candidates = target[
        "drugAndClinicalCandidates"
    ]

    if not drug_candidates:
        return []

    rows = drug_candidates["rows"]

    return rows[:max_drugs]


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("Open Targets Parkinson's Candidate Generator")
    print("=" * 60)

    # -----------------------------------------------------
    # STEP 1
    # Find Parkinson disease ID
    # -----------------------------------------------------

    print(
        f"\nLooking up disease: {DISEASE_NAME}"
    )

    disease_id, disease_name = find_disease_efo_id(
        DISEASE_NAME
    )

    print(
        f"Found: {disease_name} ({disease_id})"
    )


    # -----------------------------------------------------
    # STEP 2
    # Get Parkinson-associated targets
    # -----------------------------------------------------

    print(
        f"\nFetching top {TOP_N_TARGETS} "
        "Parkinson-associated targets..."
    )

    targets = get_top_targets(
        disease_id,
        TOP_N_TARGETS
    )

    print(
        f"Found {len(targets)} targets."
    )


    # -----------------------------------------------------
    # STEP 3
    # Retrieve drugs for each target
    # -----------------------------------------------------

    candidates = []

    for index, row in enumerate(targets, start=1):

        target = row["target"]

        target_id = target["id"]
        symbol = target["approvedSymbol"]
        target_name = target["approvedName"]

        score = row["score"]

        print(
            f"\n[{index}/{len(targets)}] "
            f"Target: {symbol}"
        )

        print(
            f"Target name: {target_name}"
        )

        print(
            f"Association score: {score:.4f}"
        )

        print(
            "Fetching drug candidates..."
        )

        drugs = get_known_drugs(
            target_id,
            MAX_DRUGS_PER_TARGET
        )

        print(
            f"Retrieved {len(drugs)} drugs."
        )


        # -------------------------------------------------
        # STEP 4
        # Store each drug-target relationship
        # -------------------------------------------------

        for drug_record in drugs:

            drug = drug_record["drug"]

            if not drug:
                continue

            candidates.append({

                "target_symbol":
                    symbol,

                "target_name":
                    target_name,

                "target_ensembl_id":
                    target_id,

                "disease_association_score":
                    score,

                "drug_name":
                    drug["name"],

                "drug_id":
                    drug["id"],

                "max_clinical_stage":
                    drug_record[
                        "maxClinicalStage"
                    ]

            })


        # Small delay between API requests
        time.sleep(0.3)


    # -----------------------------------------------------
    # STEP 5
    # Check whether candidates were found
    # -----------------------------------------------------

    if not candidates:

        print(
            "\nNo drug candidates were retrieved."
        )

        return


    # -----------------------------------------------------
    # STEP 6
    # Remove exact duplicate rows
    # -----------------------------------------------------

    unique_candidates = []

    seen = set()

    for candidate in candidates:

        key = (
            candidate["target_ensembl_id"],
            candidate["drug_id"]
        )

        if key not in seen:

            seen.add(key)
            unique_candidates.append(
                candidate
            )

    candidates = unique_candidates


    # -----------------------------------------------------
    # STEP 7
    # Save CSV
    # -----------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "target_symbol",
            "target_name",
            "target_ensembl_id",
            "disease_association_score",
            "drug_name",
            "drug_id",
            "max_clinical_stage"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(candidates)


    # -----------------------------------------------------
    # STEP 8
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print(
        f"Saved {len(candidates)} unique "
        f"drug-target rows."
    )

    print(
        f"Output file: {OUTPUT_CSV}"
    )

    print("=" * 60)

    print(
        "\nOpen Targets candidate generation complete."
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()