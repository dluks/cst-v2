#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --output=train_models_%j_%a.log
#SBATCH --error=train_models_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=190
#SBATCH --mem=500G
#SBATCH --partition=Genoa
#SBATCH --array=0-30  # Adjust based on number of traits

# Load required modules
module load enroot

# Set environment variables for optimal CPU performance
export OMP_NUM_THREADS=190
export MKL_NUM_THREADS=190
export OPENBLAS_NUM_THREADS=190
export LOKY_MAX_CPU_COUNT=190

# Create a temporary directory for the container
TMP_DIR=$(mktemp -d)
cd $TMP_DIR

# Copy the Dockerfile and project files
cp /path/to/your/project/Dockerfile .
cp -r /path/to/your/project/* .

# Build the container
enroot create --force cit-sci-traits.sqsh <<< $(docker build -q .)

# Start the container with array index to specify which trait to train
# and trait sets to use (splot, gbif, or splot_gbif)
enroot start --mount /path/to/your/data:/data \
             --mount /path/to/your/models:/models \
             cit-sci-traits.sqsh \
             --trait-index $SLURM_ARRAY_TASK_ID \
             --trait-sets splot_gbif \
             --resume \
             --debug

# Clean up
cd -
rm -rf $TMP_DIR 