import argparse, re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

ROOT = Path("/data/abar808/mllm-diagram-exp")

def normalize_answer(ans: str) -> str:
    if not ans:
        return ""
    s = ans.strip()
    m = re.search(r"Answer\s*:\s*(.+)$", s, flags=re.I | re.M)
    if m:
        s = m.group(1).strip()
    m = re.search(r"\b([ABCD])\b", s, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"-?\d+(?:\.\d+)?\s*(?:°|deg|cm|m|mm|s|kg)?", s)
    if m:
        return m.group(0).strip()
    return s.split()[0] if s else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diagram_first", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"])
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    sys_prompt = (ROOT/"prompts"/("diagram_first.txt" if args.diagram_first else "default.txt")).read_text().strip()

    print(f"[INFO] Loading {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, device_map=args.device_map
    )

    df = pd.read_csv(args.csv)
    outs = []

    for i, r in df.iterrows():
        img = Image.open(ROOT / r["image_path"]).convert("RGB")
        q = str(r["question_text"])
        choices = str(r.get("choices_optional","")).strip()
        if choices:
            q += f"\nOptions: {choices}"
        q += "\nRespond with only the final answer."

        # 1) Chat messages in LIST style (image + text)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": q},
            ]},
        ]

        # 2) Build chat string that contains the <image> placeholder.
        #    IMPORTANT: tokenize=False here.
        chat_str = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # 3) Let the processor create *aligned* input_ids + pixel_values.
        inputs = processor(text=chat_str, images=img, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )

        # Decode only newly generated tokens
        # (processor.tokenizer needed to compute prompt length)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = gen_ids[:, prompt_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        outs.append({
            "item_id": r["item_id"],
            "model_raw": text,
            "model_norm": normalize_answer(text),
            "answer_gold_diagram": str(r["answer_gold_diagram"]),
            "answer_gold_text_misleading": str(r.get("answer_gold_text_misleading","")),
            "condition": r["condition"],
        })
        if (i+1) % 25 == 0:
            print(f"[{i+1}/{len(df)}] ...")

    out_df = pd.DataFrame(outs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()
