import pandas as pd
 
PREDICTIONS_PATH = "probabilities.tsv"   # <-- update to your downloaded file
PARKINSONS_DOID = "DOID:14330"           # Disease Ontology ID for Parkinson's disease
PARKINSONS_NAME_FALLBACK = "parkinson"   # used if disease_id column isn't DOID-based
OUTPUT_CSV = "hetionet_parkinsons_candidates.csv"
TOP_N = 50
 
# If the downloaded file uses different column names, map them here:
COLUMN_MAP = {
    "compound_id": "compound_id",
    "compound_name": "compound_name",
    "disease_id": "disease_id",
    "disease_name": "disease_name",
    "prediction": "prediction",
}
 
 
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {missing} not found in {path}. "
            f"Actual columns are: {list(df.columns)}. "
            "Update COLUMN_MAP at the top of this script to match."
        )
    return df
 
 
def filter_parkinsons(df: pd.DataFrame) -> pd.DataFrame:
    disease_id_col = COLUMN_MAP["disease_id"]
    disease_name_col = COLUMN_MAP["disease_name"]
 
    mask = df[disease_id_col].astype(str).str.contains(PARKINSONS_DOID, case=False, na=False)
    if not mask.any():
        # fall back to matching on disease name if DOID format differs
        mask = df[disease_name_col].astype(str).str.contains(
            PARKINSONS_NAME_FALLBACK, case=False, na=False
        )
    return df[mask].copy()
 
 
def main():
    print(f"Loading predictions from {PREDICTIONS_PATH} ...")
    df = load_predictions(PREDICTIONS_PATH)
    print(f"Loaded {len(df):,} total compound-disease predictions")
 
    pd_candidates = filter_parkinsons(df)
    print(f"Found {len(pd_candidates)} Parkinson's-related candidates")
 
    pred_col = COLUMN_MAP["prediction"]
    pd_candidates = pd_candidates.sort_values(pred_col, ascending=False)
 
    top = pd_candidates.head(TOP_N)
    top.to_csv(OUTPUT_CSV, index=False)
 
    print(f"Saved top {len(top)} candidates (by predicted probability) to {OUTPUT_CSV}")
    print(top[[COLUMN_MAP["compound_name"], pred_col]].to_string(index=False))
 
 
if __name__ == "__main__":
    main()
 


