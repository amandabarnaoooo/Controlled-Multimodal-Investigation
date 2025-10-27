#!/bin/bash
#SBATCH --job-name=mathverse_job        # Job name
#SBATCH --output=mathverse_output.log   # Output file (%j expands to job ID)
#SBATCH --error=mathverse_error.log     # Error file
#SBATCH --time=10:00:00                 # Wall time (hh:mm:ss)
#SBATCH --gres=gpu:1                    # [REQUIRED] Request 1 GPU, If you requests more than 1, you need to get approval from the Slack channel bot)


source $HOME/miniconda/etc/profile.d/conda.sh
conda activate python3.10_env

# Move to your project directory
cd /data/ecau171/mathverse_proj
# Set path to your zip file
ZIP_FILE="/data/ecau171/mathverse_proj/mathverse_generated_*.zip"
OUTPUT_DIR="/data/ecau171/mathverse_proj/generated_images"

# Unzip the latest generated zip file
latest_zip=$(ls -t /data/ecau171/mathverse_proj/mathverse_generated_*.zip | head -n 1)
echo "Extracting latest zip: $latest_zip"
mkdir -p $OUTPUT_DIR
unzip -o "$latest_zip" -d "$OUTPUT_DIR"

# List the contents to confirm images exist
echo "Contents of generated image folder:"
ls -lh $OUTPUT_DIR

# Optionally, show the first few file names
echo "First few generated images:"
ls -1 $OUTPUT_DIR | head

# Create a small HTML gallery you can view in VS Code
HTML_FILE="$OUTPUT_DIR/index.html"
echo "<html><body><h1>Generated Images</h1>" > $HTML_FILE
for img in $OUTPUT_DIR/*.png; do
    echo "<img src=\"$(basename $img)\" width=\"256\">" >> $HTML_FILE
done
echo "</body></html>" >> $HTML_FILE

echo "✅ HTML preview created at: $HTML_FILE"
echo "You can open it in VS Code to visually inspect the generated images."