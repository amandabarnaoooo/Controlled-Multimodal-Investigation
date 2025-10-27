#!/bin/bash
#SBATCH --job-name=qwen_contra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/abar808/mllm-diagram-exp/logs/qwen_contra.out
#SBATCH --error=/data/abar808/mllm-diagram-exp/logs/qwen_contra.err
#SBATCH --open-mode=truncate

export HF_HOME=/data/abar808/mllm-diagram-exp/.hfhome

source /data/abar808/env_mllm/bin/activate
cd /data/abar808/mllm-diagram-exp

python scripts/run_inference_qwen_hf.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --csv data/build/contradictory.csv \
  --out runs/qwen_IpTperp.csv
