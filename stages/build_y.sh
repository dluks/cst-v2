#!/bin/bash

# Function to check if we're in a Slurm environment
is_slurm_available() {
    command -v sbatch >/dev/null 2>&1
}

# Function to check if container should be used
use_container() {
    [[ "${USE_CONTAINER:-FALSE}" =~ ^(TRUE|true|T|t|1)$ ]]
}

# Get the command (last argument)
ARGS=("$@")
COMMAND="${ARGS[${#ARGS[@]}-1]}"

# Main logic
if is_slurm_available; then
    echo "Running in Slurm environment..."

    # Base parameters for job submission
    PARAMS=(
        --job-name="build_y" \
        --output="logs/build_y/%j.log" \
        --error="logs/build_y/%j.err" \
        --time="00:30:00" \
        --nodes=1 \
        --ntasks=1 \
        --cpus=10 \
        --mem="600G" \
        --partition="cpu"
    )
    
    # Use the enhanced utility script with the wrapper
    stages/utils/slurm_submit.sh "${PARAMS[@]}" "stages/utils/run_in_container.sh $@"
else
    if use_container; then
        echo "Running with container..."
        stages/utils/run_in_container.sh "$@"
    else
        echo "Running directly without container..."
        # Execute the command directly
        exec $COMMAND
    fi
fi 