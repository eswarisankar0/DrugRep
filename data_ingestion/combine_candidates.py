 
import re
import pandas as pd
 
OPEN_TARGETS_CSV = "parkinsons_candidates.csv"
HETIONET_CSV = "hetionet_parkinsons_candidates.csv"
OUTPUT_CSV = "combined_candidates.csv"
 
# Common salt/form suffixes that make the *same* compound show up under
# different names across sources (e.g. Open Targets returns
# "AMANTADINE HYDROCHLORIDE" while Hetionet has plain "Amantadine").
# Stripping these lets us match on the underlying active compound.
SALT_SUFFIXES = [
    "hydrochloride", "dihydrochloride", "hcl", "sulfate", "sulphate",
    "mesylate", "tartrate", "citrate", "maleate", "fumarate", "acetate",
    "phosphate", "succinate", "besylate", "hydrobromide", "bromide",
    "sodium", "potassium", "calcium", "hydrate", "dihydrate", "anhydrous",
]
_SALT_PATTERN = re.compile(
    r"\b(" + "|".join(SALT_SUFFIXES) + r")\b", flags=re.IGNORECASE
)
 
 
def normalize_drug_name(name: str) -> str:
    """Lowercase, strip whitespace, and drop salt/form suffixes so the
    same active compound matches across sources regardless of how each
    source formatted the name."""
    name = str(name).lower().strip()
    name = _SALT_PATTERN.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name
 
 
def load_open_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "drug_name": df["drug_name"],
        "normalized_name": df["drug_name"].apply(normalize_drug_name),
        "source": "open_targets",
        "source_score": None,          # Open Targets doesn't give one combined score here
        "target_symbol": df["target_symbol"],
        "disease_association_score": df["disease_association_score"],
        "max_clinical_stage": df.get("max_clinical_stage"),
    })
    return out
 
 
def load_hetionet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    compound_col = "compound_name" if "compound_name" in df.columns else df.columns[0]
    pred_col = "prediction" if "prediction" in df.columns else df.columns[-1]
    out = pd.DataFrame({
        "drug_name": df[compound_col],
        "normalized_name": df[compound_col].apply(normalize_drug_name),
        "source": "hetionet",
        "source_score": df[pred_col],
        "target_symbol": None,
        "disease_association_score": None,
        "max_clinical_stage": None,
    })
    return out
 
 
def main():
    frames = []
 
    try:
        ot = load_open_targets(OPEN_TARGETS_CSV)
        frames.append(ot)
        print(f"Loaded {len(ot)} rows from Open Targets ({OPEN_TARGETS_CSV})")
    except FileNotFoundError:
        print(f"Skipping Open Targets: {OPEN_TARGETS_CSV} not found")
 
    try:
        het = load_hetionet(HETIONET_CSV)
        frames.append(het)
        print(f"Loaded {len(het)} rows from Hetionet ({HETIONET_CSV})")
    except FileNotFoundError:
        print(f"Skipping Hetionet: {HETIONET_CSV} not found")
 
    if not frames:
        raise SystemExit("No source CSVs found. Run the two source scripts first.")
 
    combined = pd.concat(frames, ignore_index=True)
 
    # Flag drugs that appear in both sources - these deserve extra attention,
    # since independent methods (network prediction + target/pathway biology)
    # agree on them. Match on normalized_name (salt-stripped, lowercased)
    # rather than raw drug_name so e.g. "Amantadine" and "AMANTADINE
    # HYDROCHLORIDE" are recognized as the same compound.
    counts = combined.groupby("normalized_name")["source"].nunique().rename("n_sources")
    combined = combined.merge(counts, on="normalized_name", how="left")
 
    # Sort: candidates flagged by both sources first, then by whichever
    # score is available.
    combined["sort_score"] = combined["source_score"].fillna(
        combined["disease_association_score"]
    ).fillna(0)
    combined = combined.sort_values(
        by=["n_sources", "sort_score"], ascending=[False, False]
    )
 
    combined.to_csv(OUTPUT_CSV, index=False)
 
    n_both = (combined["n_sources"] == 2).sum()
    print(f"\nSaved {len(combined)} combined rows to {OUTPUT_CSV}")
    print(f"{n_both} rows correspond to drugs flagged by BOTH sources - review these first.")
 
 
if __name__ == "__main__":
    main()
 