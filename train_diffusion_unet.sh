#!/bin/bash
#SBATCH --job-name=mv-diffusion
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=logs/train_diffusion-%j.out
#SBATCH --error=logs/train_diffusion-%j.err

export UPI=abar808
source /data/$UPI/env_mllm/bin/activate
cd /data/$UPI/mllm-diagram-exp
export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs outputs/diffusion/samples outputs/diffusion/checkpoints

python scripts/train_diffusion_unet.py \
  --data_dir data/mathverse_images \
  --out_dir outputs/diffusion \
  --image_size 64 \
  --batch_size 64 \
  --workers 4 \
  --lr 1e-4 \
  --epochs 30 \
  --timesteps 400 \
  --log_every 50 \
  --sample_every 500 \
  --save_every 1000
