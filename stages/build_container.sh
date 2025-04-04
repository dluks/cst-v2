#!/bin/bash

# Get container parameters from command line arguments
container_type=$1
container_name=$2
container_tag=$3
container_def=$4
container_image=$5

# Build container based on type
if [ "$container_type" = "docker" ]; then
    docker build -t "$container_name:$container_tag" -f "$container_def" .
else
    singularity build "$container_image" "$container_def"
fi 