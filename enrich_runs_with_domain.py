#!/usr/bin/env python3
"""
Merge domain/subcategory/source into each runs/*.csv by item_id.
Writes runs/*_with_domain.csv and prints match stats.
"""

from pathlib import Path
import pandas as pd

ROOT  = Path("/data/abar808/mllm-diagram-exp")
RUNS  = ROOT / "runs"
BUILD = ROOT / "data" / "build"
META  = BUILD / "metadata_domains.csv"

def main():
    md = pd.read_csv(META)
    if "item_id" not in md.columns:
        raise SystemExit("metadata_domains.csv must have an 'item_id' column.")
    md = md.drop_duplicates(subset=["item_id"], keep="first")

    csvs = sorted(RUNS.glob("*.csv"))
    if not csvs:
        print("[WARN] no CSVs in runs/")
        return

    for rp in csvs:
        df = pd.read_csv(rp)
        if "item_id" not in df.columns:
            print(f"[SKIP] {rp.name} has no 'item_id' column")
            continue

        merged = df.merge(md, on="item_id", how="left")
        matched = merged["domain"].notna().sum()
        total   = len(merged)
        print(f"[OK] {rp.name}: matched {matched}/{total} ({matched/total:.1%})")

        outp = rp.with_name(rp.stem + "_with_domain.csv")
        merged.to_csv(outp, index=False)
        print(f"[OK] wrote {outp}")

if __name__ == "__main__":
    main()
