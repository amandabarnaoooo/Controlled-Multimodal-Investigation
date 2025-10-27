# /data/abar808/mllm-diagram-exp/scripts/prepare_mathverse.py
"""
Prepare MathVerse samples in three conditions for 'diagram vs text' experiments.

Outputs (to /data/abar808/mllm-diagram-exp/data/build/):
- master_original.csv     (I+T)        : original question text + image
- contradictory.csv       (I+T_perp)   : add misleading "text hint" line
- masked.csv              (I+T_masked) : digits in text replaced by [NUM]
- seed_from_mathverse_<config>_<N>.csv : raw normalized rows for auditing/merging

Images are saved under:
- /data/abar808/mllm-diagram-exp/data/mathverse_images/<item_id>.png
"""

import argparse
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from PIL import Image

# ---------- Adjust if your root path differs ----------
ROOT = Path("/data/abar808/mllm-diagram-exp")
# ------------------------------------------------------

IMG_DIR = ROOT / "data" / "mathverse_images"
BUILD_DIR = ROOT / "data" / "build"
RAW_DIR = ROOT / "data" / "mathverse_raw"
for d in (IMG_DIR, BUILD_DIR, RAW_DIR):
    d.mkdir(parents=True, exist_ok=True)


def mask_numerals(text: str) -> str:
    """Replace integers/floats with [NUM]; keep surrounding units/symbols."""
    return re.sub(r"\d+(?:\.\d+)?", "[NUM]", str(text))


def choose_misleading_answer(example: Dict[str, Any]) -> str:
    """
    Pick a plausible incorrect answer for contradictory text:
    - If multiple choice exists, pick any option != gold.
    - If gold is numeric, perturb by +20% (or +1 if |val|<1).
    - Fallback: "None".
    """
    gold = str(example.get("answer", "")).strip()

    # Prefer a wrong option from choices/options/candidates
    for key in ("choices", "options", "candidates"):
        if key in example and isinstance(example[key], (list, tuple)) and example[key]:
            for o in example[key]:
                s = str(o).strip()
                if s and s != gold:
                    return s
            break

    # If numeric gold, perturb deterministically
    if re.fullmatch(r"-?\d+(?:\.\d+)?", gold):
        try:
            val = float(gold)
            new = val + 1.0 if abs(val) < 1 else val * 1.2
            if "." in gold:
                out = f"{new:.2f}".rstrip("0").rstrip(".")
            else:
                out = str(int(round(new)))
            if out == gold:
                out = str(val + 2.0)
            return out
        except Exception:
            pass

    return "None"


def save_image(loc: Path, pil_img: Image.Image) -> None:
    loc.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(loc)


def normalize_example(ex: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    Normalize a HF row to a common schema:
    - item_id, question, answer, choices (optional), image (PIL)
    Return None if essential fields are missing.
    """
    # question
    question = None
    for k in ("question", "question_text", "prompt", "instruction"):
        if k in ex and isinstance(ex[k], str) and ex[k].strip():
            question = ex[k].strip()
            break

    # answer
    answer = None
    for k in ("answer", "final_answer", "label", "solution"):
        if k in ex and isinstance(ex[k], (str, int, float)):
            answer = str(ex[k]).strip()
            break

    # image
    image = None
    for k in ("image", "img", "diagram"):
        if k in ex:
            image = ex[k]
            break

    if question is None or answer is None or image is None:
        return None

    # choices (optional)
    choices = None
    for k in ("choices", "options", "candidates"):
        if k in ex and isinstance(ex[k], (list, tuple)):
            choices = [str(x) for x in ex[k]]
            break

    item_id = ex.get("problem_id", ex.get("id", f"mv_{idx}"))
    return {
        "item_id": str(item_id),
        "question": question,
        "answer": answer,
        "choices": choices,
        "image": image,  # PIL.Image.Image expected by datasets
    }


def build_condition_csvs(rows: List[Dict[str, Any]]) -> None:
    """
    Build and write the three experiment CSVs from normalized rows.
    """
    df = pd.DataFrame(rows)

    # (I+T) original
    ipt = df.copy()
    ipt["condition"] = "I+T"
    ipt["question_text"] = ipt["question_text_original"]
    ipt["gold_correct_by"] = "diagram"

    # (I+T_perp) contradictory: append a misleading note line
    def inject_note(txt: str, mis: str) -> str:
        mis_str = str(mis) if mis is not None else "None"
        return f"{txt}\nNote: (text hint) The answer is {mis_str}."
    contra = df.copy()
    contra["condition"] = "I+T_perp"
    contra["question_text"] = [
        inject_note(t, m)
        for t, m in zip(contra["question_text_original"], contra["answer_gold_text_misleading"])
    ]
    contra["gold_correct_by"] = "diagram"

    # (I+T_masked) digits masked
    masked = df.copy()
    masked["condition"] = "I+T_masked"
    masked["question_text"] = masked["question_text_original"].apply(mask_numerals)
    masked["gold_correct_by"] = "diagram"

    # Write all three
    (BUILD_DIR / "master_original.csv").write_text(ipt.to_csv(index=False))
    (BUILD_DIR / "contradictory.csv").write_text(contra.to_csv(index=False))
    (BUILD_DIR / "masked.csv").write_text(masked.to_csv(index=False))

    print("[OK] Wrote CSVs:")
    print(" -", BUILD_DIR / "master_original.csv")
    print(" -", BUILD_DIR / "contradictory.csv")
    print(" -", BUILD_DIR / "masked.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="testmini",
                        help="HF dataset config name (e.g., 'testmini'). Avoid *_text_only.")
    parser.add_argument("--split", default=None,
                        help="Split name (auto-detected if omitted).")
    parser.add_argument("--n", type=int, default=200,
                        help="Max number of items to pull (script stops at dataset length).")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)

    # Validate config
    configs = get_dataset_config_names("AI4Math/MathVerse")
    if args.config not in configs:
        raise ValueError(f"Config '{args.config}' not found. Available: {configs}")

    # Discover splits for the config
    try:
        splits = get_dataset_split_names("AI4Math/MathVerse", args.config)
    except Exception:
        splits = []

    if not splits:
        # Single table; load and choose first key if dict
        print(f"[INFO] Loading MathVerse (config={args.config}) with NO split (single table).")
        ds_any = load_dataset("AI4Math/MathVerse", name=args.config)
        if isinstance(ds_any, dict):
            first_key = next(iter(ds_any.keys()))
            ds = ds_any[first_key]
        else:
            ds = ds_any
    else:
        chosen = args.split if (args.split in splits) else splits[0]
        print(f"[INFO] Loading MathVerse (config={args.config}, split={chosen})…")
        ds = load_dataset("AI4Math/MathVerse", name=args.config, split=chosen)

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        norm = normalize_example(ex, i)
        if norm is None:
            continue

        # Save image
        item_id = norm["item_id"]
        img_path = IMG_DIR / f"{item_id}.png"
        try:
            pil_img: Image.Image = norm["image"]
            save_image(img_path, pil_img)
        except Exception as e:
            print(f"[WARN] Skipping {item_id}: image save error: {e}")
            continue

        # Build base row
        row = {
            "item_id": item_id,
            "image_path": str(img_path.relative_to(ROOT)),  # relative to project root
            "question_text_original": norm["question"],
            "answer_gold_diagram": norm["answer"],           # assume benchmark gold aligns to diagram
            "choices_optional": ",".join(norm["choices"]) if norm["choices"] else "",
            "roi_hint_optional": "",                         # not used (for future ROI grounding)
        }
        # Choose misleading text answer
        row["answer_gold_text_misleading"] = choose_misleading_answer(
            {"answer": norm["answer"], "choices": norm["choices"]}
        )
        rows.append(row)

        if len(rows) >= args.n:
            break

    if not rows:
        print("[ERROR] No usable items found. Try a different config or increase --n.")
        return

    # Write seed for auditing/merging
    seed_csv = BUILD_DIR / f"seed_from_mathverse_{args.config}_{len(rows)}.csv"
    pd.DataFrame(rows).to_csv(seed_csv, index=False)
    print("Seed saved:", seed_csv)

    # Build experiment CSVs
    build_condition_csvs(rows)


if __name__ == "__main__":
    main()
