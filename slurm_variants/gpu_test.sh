#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=gpu-test
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1                # request exactly 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --open-mode=append
#SBATCH --output=/data/abar808/mllm-diagram-exp/logs/gpu_test.out
#SBATCH --error=/data/abar808/mllm-diagram-exp/logs/gpu_test.err

# ======== Your environment on the compute node ========
export UPI=abar808
export TMPDIR=/data/$UPI
export HOME=/data/$UPI
export HF_HOME=/data/$UPI/.cache/hf
export HUGGINGFACE_HUB_CACHE=/data/$UPI/.cache/huggingface
export TRANSFORMERS_CACHE=/data/$UPI/.cache/huggingface
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128

# activate your /data-based venv
source /data/$UPI/env_mllm/bin/activate

# run from project root (good practice)
cd /data/$UPI/mllm-diagram-exp

# run the test
python scripts/gpu_smoketest.py
