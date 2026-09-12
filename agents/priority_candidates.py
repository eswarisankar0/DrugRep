"""
agents/priority_candidates.py

Derives the PRIORITY_CANDIDATES set used by score_literature_hybrid.py
directly from combined_candidates.csv, instead of hardcoding drug names.

Two tiers of "priority", combined with OR logic:
  1. Dual-source hits (n_sources == 2) — these already survived the
     strongest filter: two independent methods agree on them.
  2. Top N by sort_score within each source — catches strong single-source
     candidates that deserve attention even without dual-source backing.

Usage:
    from priority_candidates import get_priority_candidates
    PRIORITY_CANDIDATES = get_priority_candidates()
"""

import pandas as pd

COMBINED_CSV_PATH = "../data_ingestion/combined_candidates.csv"  # adjust relative path as needed
TOP_N_PER_SOURCE = 10  # how many top single-source candidates to also include


def get_priority_candidates(csv_path: str = COMBINED_CSV_PATH,
                              top_n_per_source: int = TOP_N_PER_SOURCE) -> set:
    df = pd.read_csv(csv_path)

    # Tier 1: anything flagged by both sources
    dual_source = set(df.loc[df["n_sources"] == 2, "normalized_name"])

    # Tier 2: top N candidates per source by sort_score, even if single-source
    top_per_source = set()
    for source_name, group in df.groupby("source"):
        top = group.sort_values("sort_score", ascending=False).head(top_n_per_source)
        top_per_source.update(top["normalized_name"])

    priority = dual_source | top_per_source
    return priority


def explain_priority_candidates(csv_path: str = COMBINED_CSV_PATH,
                                  top_n_per_source: int = TOP_N_PER_SOURCE):
    """Print out WHY each candidate made the priority list — useful for
    sanity-checking before you commit LLM budget to them."""
    df = pd.read_csv(csv_path)
    priority = get_priority_candidates(csv_path, top_n_per_source)

    print(f"Priority candidates ({len(priority)} total):\n")
    for name in sorted(priority):
        rows = df[df["normalized_name"] == name]
        n_src = rows["n_sources"].iloc[0]
        sources = rows["source"].unique().tolist()
        reason = "dual-source match" if n_src == 2 else f"top-{top_n_per_source} in {sources}"
        print(f"  {name:20s} -> {reason}")


if __name__ == "__main__":
    explain_priority_candidates()