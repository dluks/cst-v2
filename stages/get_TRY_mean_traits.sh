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
    # Use the utility script with named parameters for better readability
    stages/utils/slurm_submit.sh \
        --job-name="try_traits" \
        --output="logs/try_traits/%j.log" \
        --error="logs/try_traits/%j.err" \
        --time="00:05:00" \
        --nodes=1 \
        --ntasks=1 \
        --cpus=2 \
        --mem="3G" \
        --partition="cpu" \
        "stages/utils/run_in_container.sh $@"
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
