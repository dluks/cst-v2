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

get_stats() {
    # Extract stats from config
    python -c "from src.conf.conf import get_config; print(','.join([str(t) for t in get_config().datasets.Y.trait_stats]))"
}

is_fd_mode() {
    # Check if FD mode is enabled and return proper boolean
    if python -c "from src.conf.conf import get_config; print(get_config().datasets.Y.fd_mode)" | grep -q "true"; then
        return 0  # true
    else
        return 1  # false
    fi
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
echo "Starting parallel sPlot maps generation..."

# Get resume flag if provided
RESUME=""
if [[ "$1" == "-r" || "$1" == "--resume" ]]; then
    RESUME="--resume"
    shift
fi

# Get trait flag if provided
TRAIT=""
if [[ "$1" == "-t" || "$1" == "--trait" ]]; then
    TRAIT="--trait $2"
    shift 2
fi

# Get all trait IDs if no specific trait was provided
if is_fd_mode; then
    STATS=$(get_stats)
    IFS=',' read -ra STAT_ARRAY <<< "$STATS"
    echo "Found ${#STAT_ARRAY[@]} stats to process"
elif [[ -z "$TRAIT" ]]; then
    TRAITS=$(get_traits)
    IFS=',' read -ra TRAIT_ARRAY <<< "$TRAITS"
    echo "Found ${#TRAIT_ARRAY[@]} traits to process"
else
    # Extract the trait number from the --trait argument
    TRAIT_NUM=$(echo "$TRAIT" | sed -e 's/--trait //')
    TRAIT_ARRAY=("$TRAIT_NUM")
    echo "Processing single trait: $TRAIT_NUM"
fi

if is_slurm_available; then
    echo "Running in Slurm environment..."
    
    # Base parameters for job submission - reduced resources per job since we're processing one trait at a time
    PARAMS=(
        --output="logs/build_splot_maps/%j_%a.log" \
        --error="logs/build_splot_maps/%j_%a.err" \
        --time="02:00:00" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=80 \
        --threads-per-core=1 \
        --mem="50G" \
        --partition="cpu"
    )
    
    if is_fd_mode; then
        for STAT in "${STAT_ARRAY[@]}"; do
            JOB_NAME="sp_${STAT}"
            echo "Submitting job for stat ${STAT}..."
            
            # Construct command to run for this stat
            if use_container; then
                CMD="stages/utils/run_in_container.sh $CONTAINER_TYPE $CONTAINER_NAME $CONTAINER_TAG $CONTAINER_IMAGE poetry run python -m src.data.build_splot_maps ${RESUME} --fd-metric ${STAT}"
            else
                CMD="poetry run python -m src.data.build_splot_maps ${RESUME} --fd-metric ${STAT}"
            fi
            
            # Submit job
            sbatch --job-name="${JOB_NAME}" "${PARAMS[@]}" --wrap="${CMD}"
        done
    else
        for TRAIT in "${TRAIT_ARRAY[@]}"; do
            # Create job name with trait ID
            JOB_NAME="sp_${TRAIT}"
            
            echo "Submitting job for trait ${TRAIT}..."
            
            # Construct command to run for this trait
            if use_container; then
                CMD="stages/utils/run_in_container.sh $CONTAINER_TYPE $CONTAINER_NAME $CONTAINER_TAG $CONTAINER_IMAGE poetry run python -m src.data.build_splot_maps ${RESUME} --trait ${TRAIT}"
            else
                CMD="poetry run python -m src.data.build_splot_maps ${RESUME} --trait ${TRAIT}"
            fi
            
            # Submit job
            sbatch --job-name="${JOB_NAME}" "${PARAMS[@]}" --wrap="${CMD}"
        done
    fi
    
    echo "All trait rasterization jobs submitted"
else
    echo "Not running in Slurm environment, processing sequentially..."
    
    if is_fd_mode; then
        for STAT in "${STAT_ARRAY[@]}"; do
            echo "Processing stat ${STAT}..."
            
            if use_container; then
                stages/utils/run_in_container.sh $CONTAINER_TYPE $CONTAINER_NAME $CONTAINER_TAG $CONTAINER_IMAGE poetry run python -m src.data.build_splot_maps ${RESUME} --fd-metric ${STAT}
            else
                poetry run python -m src.data.build_splot_maps ${RESUME} --fd-metric ${STAT}
            fi
        done
    else
        for TRAIT in "${TRAIT_ARRAY[@]}"; do
            echo "Processing trait ${TRAIT}..."
            
            if use_container; then
                stages/utils/run_in_container.sh $CONTAINER_TYPE $CONTAINER_NAME $CONTAINER_TAG $CONTAINER_IMAGE poetry run python -m src.data.build_splot_maps ${RESUME} --trait ${TRAIT}
            else
                poetry run python -m src.data.build_splot_maps ${RESUME} --trait ${TRAIT}
            fi
        done
    fi
fi 