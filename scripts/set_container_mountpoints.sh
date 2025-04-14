#!/bin/bash

# Create the bind path string with additional MPI-related paths
BIND_PATHS="src:/app/src,tests:/app/tests,models:/app/models,data:/app/data,reference:/app/reference,results:/app/results,profiling:/app/profiling,tmp:/app/tmp,.env:/app/.env,params.yaml:/app/params.yaml"

echo "BIND_PATHS: $BIND_PATHS"

# Set the bind path
export APPTAINER_BINDPATH="$BIND_PATHS"

echo "APPTAINER_BINDPATH: $APPTAINER_BINDPATH"

# If this script was sourced, the variable will be set in the parent shell
# If it was run directly, we need to tell the user to source it
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Warning: This script was run directly. To set the environment variable in your current shell, run:"
    echo "source $0"
    echo "or"
    echo ". $0"
fi