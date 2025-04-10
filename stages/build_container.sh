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
    # Temporarily backup and unset APPTAINER_BINDPATH if it's set, otherwise the
    # build process will fail because the bind targets don't exist yet.
    if [ -n "$APPTAINER_BINDPATH" ]; then
        echo "Temporarily unsetting APPTAINER_BINDPATH for container build"
        OLD_BINDPATH=$APPTAINER_BINDPATH
        unset APPTAINER_BINDPATH
    fi
    
    # Build with Apptainer
    apptainer build --force "$container_image" "$container_def"
    
    # Restore APPTAINER_BINDPATH if it was previously set
    if [ -n "$OLD_BINDPATH" ]; then
        export APPTAINER_BINDPATH=$OLD_BINDPATH
    fi
fi 