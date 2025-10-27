#!/bin/bash
#SBATCH --job-name=contra-llava
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=/data/abar808/mllm-diagram-exp/logs/infer_llava_contra.out
#SBATCH --error=/data/abar808/mllm-diagram-exp/logs/infer_llava_contra.err
export UPI=abar808
export TMPDIR=/data/$UPI; export HOME=/data/$UPI
export HF_HOME=/data/$UPI/.cache/hf
export HUGGINGFACE_HUB_CACHE=/data/$UPI/.cache/huggingface
export TRANSFORMERS_CACHE=/data/$UPI/.cache/huggingface
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
source /data/$UPI/env_mllm/bin/activate
cd /data/$UPI/mllm-diagram-exp
python scripts/run_inference_llava_hf.py \
  --model llava-hf/llava-1.5-7b-hf \
  --csv data/build/contradictory.csv \
  --out runs/llava15_IpTperp.csv
