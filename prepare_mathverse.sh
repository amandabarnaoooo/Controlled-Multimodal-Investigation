#!/bin/bash
#SBATCH --job-name=prep-mathverse
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/data/abar808/mllm-diagram-exp/logs/prep_mathverse.out
#SBATCH --error=/data/abar808/mllm-diagram-exp/logs/prep_mathverse.err

export UPI=abar808
export TMPDIR=/data/$UPI
export HOME=/data/$UPI
export HF_HOME=/data/$UPI/.cache/hf
export HUGGINGFACE_HUB_CACHE=/data/$UPI/.cache/huggingface
export TRANSFORMERS_CACHE=/data/$UPI/.cache/huggingface
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128

source /data/$UPI/env_mllm/bin/activate
cd /data/$UPI/mllm-diagram-exp

# pull N items from the 'test' split (adjust --n as you like)
python scripts/prepare_mathverse.py --split test --n 300
