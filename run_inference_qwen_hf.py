#!/usr/bin/env python3
"""
Run inference with Qwen2-VL on one of your prepared CSVs (IpT / masked / contra).

Usage (example):
  python scripts/run_inference_qwen_hf.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --csv   data/build/master_original.csv \
    --out   runs/qwen_IpT.csv

Notes
- Uses HF chat template with a list-style message (image + text).
- We call `apply_chat_template(..., tokenize=False)` to get a string that
  contains the special image placeholder; then we let the processor build
  aligned input_ids + pixel_values via `processor(text=..., images=...)`.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration  # model class for Qwen2-VL

ROOT = Path("/data/abar808/mllm-diagram-exp")


def normalize_answer(ans: str) -> str:
    """Heuristics to normalize the model's final answer."""
    if not ans:
        return ""
    s = str(ans).strip()
    # common "Answer: ..." patterns
    m = re.search(r"Answer\s*:\s*(.+)$", s, flags=re.I | re.M)
    if m:
        s = m.group(1).strip()
    # single-choice letter
    m = re.search(r"\b([ABCD])\b", s, flags=re.I)
    if m:
        return m.group(1).upper()
    # numeric with optional unit
    m = re.search(r"-?\d+(?:\.\d+)?\s*(?:°|deg|cm|m|mm|s|kg)?", s)
    if m:
        return m.group(0).strip()
    # fallback: first token
    return s.split()[0] if s else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct",
                    help="Hugging Face model id for Qwen2-VL.")
    ap.add_argument("--csv", required=True, help="Input CSV (IpT / masked / contra).")
    ap.add_argument("--out", required=True, help="Output predictions CSV path.")
    ap.add_argument("--diagram_first", action="store_true",
                    help="Use the diagram-first system prompt.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Load the appropriate system prompt
    prompt_file = "diagram_first.txt" if args.diagram_first else "default.txt"
    sys_prompt = (ROOT / "prompts" / prompt_file).read_text().strip()

    print(f"[INFO] Loading {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )

    df = pd.read_csv(args.csv)
    outs = []

    for i, r in df.iterrows():
        # Load image
        img_path = ROOT / r["image_path"]
        img = Image.open(img_path).convert("RGB")

        # Build the user query text
        q = str(r["question_text"])
        choices = str(r.get("choices_optional", "")).strip()
        if choices:
            q += f"\nOptions: {choices}"
        q += "\nRespond with only the final answer."

        # Qwen2-VL chat messages (LIST style with an image block)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": q},
            ]},
        ]

        # Step 1: build a chat string with special image placeholder (no tokenization here)
        chat_str = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Step 2: let the processor create *aligned* tensors (input_ids + pixel_values)
        inputs = processor(text=chat_str, images=img, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        # Decode only newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = gen_ids[:, prompt_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        outs.append({
            "item_id": r["item_id"],
            "model_raw": text,
            "model_norm": normalize_answer(text),
            "answer_gold_diagram": str(r["answer_gold_diagram"]),
            "answer_gold_text_misleading": str(r.get("answer_gold_text_misleading", "")),
            "condition": r["condition"],
        })

        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(df)}] ...")

    out_df = pd.DataFrame(outs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print("[OK] wrote", args.out)


if __name__ == "__main__":
    main()
