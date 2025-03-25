#!/bin/bash

# Extract trait IDs from params.yaml
TRAIT_IDS=$(yq e '.datasets.Y.traits[]' params.yaml)

# Create a directory for the SLURM tasks
mkdir -p slurm_tasks

# Generate a SLURM task for each trait ID
for trait_id in $TRAIT_IDS; do
    cat > "slurm_tasks/train_trait_${trait_id}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=train_${trait_id}
#SBATCH --output=logs/train_${trait_id}_%j.log
#SBATCH --error=logs/train_${trait_id}_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=190
#SBATCH --mem=100GB
#SBATCH --partition=cpu

# Load required modules
module load singularity

# Activate conda environment
source ~/.bashrc
conda activate cit-sci-traits

# Run the training script
python -m src.models.train_models --trait-id ${trait_id} --resume
EOF

    # Make the script executable
    chmod +x "slurm_tasks/train_trait_${trait_id}.slurm"
done

echo "Generated SLURM tasks for ${#TRAIT_IDS[@]} traits in slurm_tasks/" 