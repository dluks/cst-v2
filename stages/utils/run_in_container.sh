#!/bin/bash

# Function to get container command based on type
get_container_cmd() {
    local container_type=$1
    local container_name=$2
    local container_tag=$3
    local container_image=$4
    local cmd=$5

    if [ "$container_type" = "docker" ]; then
        echo "docker run --rm -v \$(pwd):/app -w /app $container_name:$container_tag $cmd"
    else
        # Set APPTAINER_BINDPATH environment variable for Apptainer/Singularity
        # This will bind all the paths defined in the config file
        export APPTAINER_BINDPATH="\
            src:/app/src,\
            tests:/app/tests,\
            models:/app/models,\
            data:/app/data,\
            reference:/app/reference,\
            results:/app/results,\
            .env:/app/.env"
        echo "apptainer run $container_image $cmd"
    fi
}

# Main logic
container_type=${1:-"singularity"}
container_name=${2:-"cit-sci-traits"}
container_tag=${3:-"latest"}
container_image=${4:-"cit-sci-traits.sif"}
shift 4

# Get any additional arguments and construct the python command
python_script="$1"
shift 1

container_cmd=$(get_container_cmd "$container_type" "$container_name" "$container_tag" "$container_image" "poetry run python")
eval "$container_cmd $python_script $@" 