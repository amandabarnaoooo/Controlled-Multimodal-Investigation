#!/bin/bash
#SBATCH --job-name=mathverse_job        # Job name
#SBATCH --output=mathverse_output_VisionOnly.log   # Output file (%j expands to job ID)
#SBATCH --error=mathverse_error_VisionOnly.log     # Error file
#SBATCH --time=10:00:00                 # Wall time (hh:mm:ss)
#SBATCH --gres=gpu:1                    # [REQUIRED] Request 1 GPU, If you requests more than 1, you need to get approval from the Slack channel bot)


source $HOME/miniconda/etc/profile.d/conda.sh
conda activate python3.10_env

# Move to your project directory
cd /data/ecau171/mathverse_proj



# Run your script
#python3 load_dataset.py
python3 diffusion_model_all.py
