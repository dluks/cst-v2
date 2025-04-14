#!/bin/bash

# Function to check if we're in a Slurm environment
is_slurm_available() {
    command -v sbatch >/dev/null 2>&1
}

# Function to check if container should be used
use_container() {
    [[ "${USE_CONTAINER:-FALSE}" =~ ^(TRUE|true|T|t|1)$ ]]
}

# Function to get traits from config
get_traits() {
    # Extract trait IDs from config
    python -c "from src.conf.conf import get_config; print(','.join([str(t) for t in get_config().datasets.Y.traits]))"
}

# Function to get container parameters from params.yaml
get_container_params() {
    # Extract container parameters using yq (assumes yq is installed)
    # If yq isn't available, hardcode the values from params.yaml
    if command -v yq >/dev/null 2>&1; then
        CONTAINER_TYPE=$(yq '.container.type' params.yaml)
        CONTAINER_NAME=$(yq '.container.name' params.yaml)
        CONTAINER_TAG=$(yq '.container.docker_tag' params.yaml)
        CONTAINER_IMAGE=$(yq '.container.image' params.yaml)
    else
        # Fallback to default values from params.yaml
        CONTAINER_TYPE="singularity"
        CONTAINER_NAME="cit-sci-traits"
        CONTAINER_TAG="latest"
        CONTAINER_IMAGE="cit-sci-traits.sif"
    fi
    
    echo "$CONTAINER_TYPE $CONTAINER_NAME $CONTAINER_TAG $CONTAINER_IMAGE"
}

# Main logic
echo "Starting parallel GBIF maps generation..."

# Get overwrite flag if provided
OVERWRITE=""
if [[ "$1" == "-o" || "$1" == "--overwrite" ]]; then
    OVERWRITE="--overwrite"
    shift
fi

# Get all trait IDs
TRAITS=$(get_traits)
IFS=',' read -ra TRAIT_ARRAY <<< "$TRAITS"

echo "Found ${#TRAIT_ARRAY[@]} traits to process"

# Get container parameters
read -r CONTAINER_TYPE CONTAINER_NAME CONTAINER_TAG CONTAINER_IMAGE <<< "$(get_container_params)"

# Set up directory for logs
mkdir -p logs/build_gbif_maps

if is_slurm_available; then
    echo "Running in Slurm environment..."
    
    # Base parameters for job submission - reduced resources per job since we're processing one trait at a time
    PARAMS=(
        --output="logs/build_gbif_maps/%j_%a.log" \
        --error="logs/build_gbif_maps/%j_%a.err" \
        --time="00:30:00" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --threads-per-core=1 \
        --mem="250G" \
        --partition="cpu"
    )
    
    for TRAIT in "${TRAIT_ARRAY[@]}"; do
        # Create job name with trait ID
        JOB_NAME="build_gbif_map_${TRAIT}"
        
        echo "Submitting job for trait ${TRAIT}..."
        
        # Construct command to run for this trait
        if use_container; then
            CMD="stages/utils/run_in_container.sh $CONTAINER_TYPE $CONTAINER_NAME $CONTAINER_TAG $CONTAINER_IMAGE poetry run python -m src.data.build_gbif_maps ${OVERWRITE} --trait ${TRAIT}"
        else
            CMD="poetry run python -m src.data.build_gbif_maps ${OVERWRITE} --trait ${TRAIT}"
        fi
        
        # Submit job
        sbatch --job-name="${JOB_NAME}" "${PARAMS[@]}" --wrap="${CMD}"
    done
    
    echo "All trait rasterization jobs submitted"
else
    echo "Not running in Slurm environment, processing sequentially..."
    
    for TRAIT in "${TRAIT_ARRAY[@]}"; do
        echo "Processing trait ${TRAIT}..."
        
        if use_container; then
            stages/utils/run_in_container.sh $CONTAINER_TYPE $CONTAINER_NAME $CONTAINER_TAG $CONTAINER_IMAGE poetry run python -m src.data.build_gbif_maps ${OVERWRITE} --trait ${TRAIT}
        else
            poetry run python -m src.data.build_gbif_maps ${OVERWRITE} --trait ${TRAIT}
        fi
    done
    
    echo "All traits processed"
fi 