#!/bin/bash
#SBATCH --job-name=setup_mathverse_env
#SBATCH --output=setup_env_output.log
#SBATCH --error=setup_env_error.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Activate the virtual environment
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate python3.10_env

# Install diffusers via pip (no conda package)
# Fix transformers/tokenizers compatibility
pip install "tokenizers<0.22,>=0.21" --upgrade
pip install --upgrade transformers


# Confirm installation
echo "Installed packages:"
pip list
