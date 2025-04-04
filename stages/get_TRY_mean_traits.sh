#!/bin/bash

# Function to check if we're in a Slurm environment
is_slurm_available() {
    command -v sbatch >/dev/null 2>&1
}

# Main logic
if is_slurm_available; then
    echo "Running in Slurm environment..."
    # Use the utility script with named parameters for better readability
    stages/utils/slurm_submit.sh \
        --job-name="try_traits" \
        --output="logs/try_traits/%j.log" \
        --error="logs/try_traits/%j.err" \
        --time="00:30:00" \
        --nodes=1 \
        --ntasks=1 \
        --cpus=8 \
        --mem="3G" \
        --partition="cpu" \
        "stages/utils/run_in_container.sh $@"
else
    echo "Running directly (no Slurm detected)..."
    stages/utils/run_in_container.sh "$@"
fi
