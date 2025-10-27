#!/usr/bin/env python3
"""
Export a mapping of item_id -> {domain, subcategory, source, image_basename}
from MathVerse (config: testmini). Handles nested 'metadata' and avoids PIL objects.

Output:
  data/build/metadata_domains.csv
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset
from PIL import Image as PILImage  # only used for type checks


# ---------- Paths ----------
ROOT = Path("/data/abar808/mllm-diagram-exp")
OUT  = ROOT / "data" / "build" / "metadata_domains.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def is_stringy(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def clean_id(*candidates) -> str:
    """
    Return the first candidate that is a usable ID:
      - non-empty string, OR
      - numeric (converted to string)
    Never return objects like PIL images.
    """
    for v in candidates:
        if isinstance(v, (int, float)):
            return str(v)
        if is_stringy(v):
            return v
    return ""


def basename(p) -> str:
    try:
        return Path(str(p)).name
    except Exception:
        return ""


# ---------- Main ----------
def main():
    # Load dataset (config: testmini)
    ds = load_dataset("AI4Math/MathVerse", "testmini")

    # Pick the proper split
    if hasattr(ds, "keys"):  # DatasetDict
        if "testmini" in ds.keys():
            split = ds["testmini"]
        elif "test" in ds.keys():
            split = ds["test"]
        else:
            raise SystemExit(f"Available splits: {list(ds.keys())}. Neither 'testmini' nor 'test' found.")
    else:
        split = ds  # single Dataset

    rows = []
    for idx, ex in enumerate(split):
        d  = dict(ex)
        md = d.get("metadata", {}) or {}

        # Prefer explicit string-ish IDs from metadata if present
        item_id = clean_id(
            d.get("item_id"),
            d.get("qid"),
            d.get("id"),
            md.get("item_id"),
            md.get("qid"),
            md.get("id"),
            md.get("question_id"),
            md.get("problem_id"),
        )
        if not item_id or isinstance(item_id, PILImage.Image):
            # fallback: stable synthetic id WITHOUT zero padding (matches mv_0, mv_1, …)
            item_id = f"mv_{idx}"

        # image basename candidates (avoid grabbing PIL objects)
        image_basename = ""
        for key in ("image_name", "image_filename", "image_file", "image_path"):
            v = md.get(key)
            if is_stringy(v):
                image_basename = basename(v)
                break

        # Category info lives inside metadata (based on your dataset preview)
        domain      = clean_id(md.get("subject"),  md.get("domain"),    md.get("category"))
        subcategory = clean_id(md.get("subfield"), md.get("subdomain"), md.get("subcategory"))
        source      = clean_id(md.get("source"),   md.get("origin"),    md.get("dataset"))

        rows.append({
            "item_id": item_id,
            "image_basename": image_basename,
            "domain": domain,
            "subcategory": subcategory,
            "source": source,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["item_id"], keep="first")
    df.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT}  rows={len(df)}")
    # quick preview
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
