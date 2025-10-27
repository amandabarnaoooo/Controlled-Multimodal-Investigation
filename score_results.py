#!/usr/bin/env python3
"""
Score one or more run CSVs (e.g., IpT, Masked, Contradictory) produced by your inference scripts.

Outputs (CSV-only):
  - <OUT>.csv                       : per-run summary with columns:
                                      model, run, condition, n, acc_vs_diagram_gold,
                                      agree_diagram_rate(contradictory_only),
                                      agree_text_rate(contradictory_only)
  - <OUT>_mcnemar.csv               : McNemar test comparing IpT vs Perp (if both present)
  - <OUT>_<run>_examples_text_shortcut.csv : up to N 'text-shortcut' example rows for qualitative analysis (only for Perp runs)

Usage examples:
  python scripts/score_results.py \
    --runs runs/llava15_IpT.csv runs/llava15_IpTperp.csv runs/llava15_IpTmasked.csv \
    --out  results/tables/summary_llava15.csv

  python scripts/score_results.py \
    --runs runs/qwen_IpT.csv runs/qwen_IpTperp.csv runs/qwen_IpTmasked.csv \
    --out  results/tables/summary_qwen.csv

  # Mix models in one summary:
  python scripts/score_results.py \
    --runs runs/llava15_IpT.csv runs/llava15_IpTperp.csv runs/qwen_IpTperp.csv \
    --out  results/tables/summary_mixed.csv
"""

import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust ROOT to your workspace if needed
ROOT = Path("/data/abar808/mllm-diagram-exp")
OUT_TABLES = ROOT / "results" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# ------------------------
# Helpers: parsing & normalization
# ------------------------

def _parse_model_and_condition_from_filename(path: Path):
    """
    Infer model name (prefix before first underscore) and condition from filename.
    Examples:
      llava15_IpTperp.csv  -> model='llava15', condition='I+T_perp'
      qwen_IpTmasked.csv   -> model='qwen',    condition='I+T_masked'
      qwen_IpT.csv         -> model='qwen',    condition='I+T'
    """
    stem = path.stem
    # model name = before first underscore, else whole stem
    if "_" in stem:
        model = stem.split("_", 1)[0]
        suffix = stem.split("_", 1)[1].lower()
    else:
        model = stem
        suffix = stem.lower()

    if "perp" in suffix or "contra" in suffix or "contradict" in suffix:
        cond = "I+T_perp"
    elif "mask" in suffix:
        cond = "I+T_masked"
    else:
        cond = "I+T"
    return model, cond

def _as_float_if_fraction(s: str):
    """Convert simple fractions (e.g., '1/2') to decimal string; otherwise return original string."""
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0:
            return f"{num/den:.6g}"
    return s

def _norm(x):
    """Normalize answers so we can compare text vs diagram numeric/option outputs."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Remove thousands separators
    s = s.replace(",", "")
    # Lone MCQ option like A/B/C/D
    m = re.search(r"\b([ABCD])\b", s, flags=re.I)
    if m:
        return m.group(1).upper()
    # Bare fraction
    s_frac = _as_float_if_fraction(s)
    if s_frac != s:
        s = s_frac
    # Numeric token (with optional sci-notation) + common units tolerated
    m = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*(?:°|deg|cm|m|mm|s|kg)?", s, flags=re.I)
    if m:
        return m.group(0).strip()
    return s

def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# Column aliases across models
ALIASES = {
    "item_id": ["item_id", "id", "question_id", "problem_id", "qid"],
    "model_raw": ["model_raw", "raw", "prediction_raw", "answer_raw", "model"],
    "model_norm": [
        "model_norm", "modelanswer", "model_answer", "prediction",
        "answer_model", "final_answer", "answer", "response"
    ],
    "answer_gold_diagram": [
        "answer_gold_diagram", "diagram_gold", "gold_diagram",
        "answer_diagram", "label_diagram", "gt_diagram"
    ],
    "answer_gold_text_misleading": [
        "answer_gold_text_misleading", "text_misleading", "misleading_text",
        "answer_gold_text", "text_gold", "gt_text"
    ],
    "condition": ["condition", "split", "setting"],
}

def _get_col(df: pd.DataFrame, key: str, default=None):
    for name in ALIASES[key]:
        if name in df.columns:
            return df[name]
    if default is None:
        return pd.Series(["" for _ in range(len(df))])
    return pd.Series([default for _ in range(len(df))])

def _condition_from_name(name: str) -> str:
    n = name.lower()
    if "perp" in n or "contra" in n or "contradict" in n:
        return "I+T_perp"
    if "mask" in n:
        return "I+T_masked"
    return "I+T"

# ------------------------
# Load a run & standardize
# ------------------------

def load_run(path: str):
    p = Path(path)
    df = pd.read_csv(p)
    df = _lower_columns(df)

    out = pd.DataFrame()
    out["item_id"] = _get_col(df, "item_id").astype(str)
    out["model_raw"] = _get_col(df, "model_raw")
    out["model_norm"] = _get_col(df, "model_norm")
    out["answer_gold_diagram"] = _get_col(df, "answer_gold_diagram")
    out["answer_gold_text_misleading"] = _get_col(df, "answer_gold_text_misleading")

    # Prefer an explicit condition column if present; else infer from filename
    cond_series = _get_col(df, "condition")
    if cond_series.eq("").all():
        out["condition"] = _condition_from_name(p.stem)
    else:
        out["condition"] = cond_series.apply(
            lambda x: _condition_from_name(str(x)) if str(x).strip() else _condition_from_name(p.stem)
        )

    # Normalize comparable fields
    for c in ["model_norm", "answer_gold_diagram", "answer_gold_text_misleading"]:
        out[c] = out[c].apply(_norm)

    # Also attach model name (from filename)
    model, cond_inferred = _parse_model_and_condition_from_filename(p)
    out["model_name_from_path"] = model
    # If condition column was empty, we already injected filename-derived one above.
    return out

# ------------------------
# McNemar paired test
# ------------------------

def mcnemar_stats(df_ipt, df_perp):
    a = df_ipt.set_index("item_id")["model_norm"]
    b = df_perp.set_index("item_id")["model_norm"]
    g = df_ipt.set_index("item_id")["answer_gold_diagram"]

    ix = a.index.intersection(b.index).intersection(g.index)
    if len(ix) == 0:
        return {"paired_items": 0, "n01": 0, "n10": 0, "mcnemar_chi2": 0.0}

    correct_a = a.loc[ix] == g.loc[ix]  # IpT correct?
    correct_b = b.loc[ix] == g.loc[ix]  # Perp correct?

    # Count discordant pairs (continuity-corrected chi^2)
    n01 = int(((+correct_a) & (~correct_b)).sum())  # IpT correct, Perp wrong
    n10 = int(((~correct_a) & (+correct_b)).sum())  # IpT wrong,  Perp correct
    num = (abs(n01 - n10) - 1) ** 2
    den = max(n01 + n10, 1)
    chi2 = num / den
    return {"paired_items": int(len(ix)), "n01": n01, "n10": n10, "mcnemar_chi2": float(chi2)}

# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="CSV(s) from runs/")
    ap.add_argument("--out", required=True, help="summary CSV to write")
    ap.add_argument("--examples-max", type=int, default=50, help="Max examples to save for text-shortcut cases")
    args = ap.parse_args()

    rows = []
    loaded = {}     # name -> df
    meta = {}       # name -> (model, condition)
    for rp in args.runs:
        p = Path(rp)
        name = p.stem
        df = load_run(rp)

        # Condition to use for scoring rows
        cond = df["condition"].iloc[0] if "condition" in df.columns else _condition_from_name(name)
        # Model name for summary (from filename)
        model_name, cond_from_file = _parse_model_and_condition_from_filename(p)
        loaded[name] = df
        meta[name] = (model_name, cond)

        n = len(df)
        acc = (df["model_norm"] == df["answer_gold_diagram"]).mean() if n > 0 else 0.0

        diag_rate = text_rate = np.nan
        if cond == "I+T_perp":
            m = df["condition"] == "I+T_perp"
            diag_rate = (df.loc[m, "model_norm"] == df.loc[m, "answer_gold_diagram"]).mean()
            has_txt = df.loc[m, "answer_gold_text_misleading"].astype(str) != ""
            text_rate = (df.loc[m & has_txt, "model_norm"] == df.loc[m & has_txt, "answer_gold_text_misleading"]).mean()

        rows.append({
            "model": model_name,
            "run": name,
            "condition": cond,
            "n": int(n),
            "acc_vs_diagram_gold": float(acc),
            "agree_diagram_rate(contradictory_only)": float(diag_rate) if not np.isnan(diag_rate) else "",
            "agree_text_rate(contradictory_only)": float(text_rate) if not np.isnan(text_rate) else "",
        })

    # Write summary CSV
    summary = pd.DataFrame(rows).sort_values(["model", "condition", "run"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print("[OK] wrote summary:", out_path)

    # McNemar if both IpT and Perp present among the provided runs
    # We look for a *pair* within the same model if available; otherwise any IpT+Perp pair.
    ipt_candidates = [k for k, (_, cond) in meta.items() if cond == "I+T"]
    perp_candidates = [k for k, (_, cond) in meta.items() if cond == "I+T_perp"]

    mcnemar_rows = []
    # Prefer model-matched pairs
    for ipt_name in ipt_candidates:
        ipt_model, _ = meta[ipt_name]
        # find a perp with same model
        same_model_perps = [k for k in perp_candidates if meta[k][0] == ipt_model]
        target_perps = same_model_perps if same_model_perps else perp_candidates
        for perp_name in target_perps:
            stats = mcnemar_stats(loaded[ipt_name], loaded[perp_name])
            stats["ipt_run"] = ipt_name
            stats["perp_run"] = perp_name
            stats["model_pair"] = f"{ipt_model} vs {meta[perp_name][0]}"
            mcnemar_rows.append(stats)

    if mcnemar_rows:
        mcnemar_df = pd.DataFrame(mcnemar_rows)
        mcnemar_path = OUT_TABLES / f"{out_path.stem}_mcnemar.csv"
        mcnemar_df.to_csv(mcnemar_path, index=False)
        print("[OK] wrote:", mcnemar_path)

    # Save examples for text-shortcut behavior from each Perp run
    for name, df in loaded.items():
        _, cond = meta[name]
        if cond == "I+T_perp" and len(df) > 0:
            m = (df["answer_gold_text_misleading"].astype(str) != "") & \
                (df["model_norm"] == df["answer_gold_text_misleading"])
            examples = df.loc[m].head(args.examples_max)
            if not examples.empty:
                p = OUT_TABLES / f"{out_path.stem}_{name}_examples_text_shortcut.csv"
                examples.to_csv(p, index=False)
                print("[OK] wrote examples:", p)

if __name__ == "__main__":
    main()
