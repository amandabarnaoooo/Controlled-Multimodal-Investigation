#!/bin/bash
#SBATCH --job-name=mathverse_job        # Job name
#SBATCH --output=mathverse_output_clean%j.log   # Output file (%j expands to job ID)
#SBATCH --error=mathverse_error_clean%j.log     # Error file
#SBATCH --time=24:00:00                 # Wall time (hh:mm:ss)
#SBATCH --gres=gpu:2                    # [REQUIRED] Request 1 GPU, If you requests more than 1, you need to get approval from the Slack channel bot)


source $HOME/miniconda/etc/profile.d/conda.sh
conda activate python3.10_env

# Move to your project directory
cd /data/ecau171/mathverse_proj



# Run your script

for i in 0 1 2 3 4
do
    echo "🚀 Starting problem version $i..."
    python3 diffusion_model_all.py $i > "mathverse_output_clean_${i}.log" 2> "mathverse_error_clean_${i}.log"
    echo "✅ Finished problem version $i"
done

