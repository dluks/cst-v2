#!/bin/bash
# Utility script for submitting jobs to Slurm and waiting for completion
# Uses named parameters for better readability

# Default values
JOB_NAME="dvc_job"
OUTPUT_PATH="logs/%x_%j.log"
ERROR_PATH="logs/%x_%j.err"
TIME="01:00:00"
NODES=1
NTASKS=1
CPUS=1
MEM="4G"
PARTITION="cpu"
COMMAND=""

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --job-name=*)
      JOB_NAME="${1#*=}"
      shift
      ;;
    --output=*)
      OUTPUT_PATH="${1#*=}"
      shift
      ;;
    --error=*)
      ERROR_PATH="${1#*=}"
      shift
      ;;
    --time=*)
      TIME="${1#*=}"
      shift
      ;;
    --nodes=*)
      NODES="${1#*=}"
      shift
      ;;
    --ntasks=*)
      NTASKS="${1#*=}"
      shift
      ;;
    --cpus=*)
      CPUS="${1#*=}"
      shift
      ;;
    --mem=*)
      MEM="${1#*=}"
      shift
      ;;
    --partition=*)
      PARTITION="${1#*=}"
      shift
      ;;
    --command=*)
      COMMAND="${1#*=}"
      shift
      ;;
    *)
      # If no named parameter is given, assume it's the command
      if [ -z "$COMMAND" ]; then
        COMMAND="$*"
        break
      else
        echo "Unknown parameter: $1"
        exit 1
      fi
      ;;
  esac
done

# Validate command
if [ -z "$COMMAND" ]; then
  echo "Error: No command specified"
  echo "Usage: $0 --job-name=<name> --output=<path> --error=<path> --time=<time> --nodes=<n> --ntasks=<n> --cpus=<n> --mem=<mem> --partition=<part> --command=<cmd>"
  echo "Or: $0 [named_params] <command>"
  exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Create a temporary script
temp_script=$(mktemp)

# Write the script header
cat > "$temp_script" << EOF
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUTPUT_PATH
#SBATCH --error=$ERROR_PATH
#SBATCH --time=$TIME
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --partition=$PARTITION

# Run the command
$COMMAND
EOF

# Make script executable
chmod +x "$temp_script"

# Submit the job
job_id=$(sbatch --parsable "$temp_script")
echo "Submitted job ${job_id}, waiting for completion..."

# Wait for the job to complete
srun --dependency=afterany:${job_id} --quiet sleep 1

# Check job status
sacct -j ${job_id} --format=State --noheader | grep -q "COMPLETED"
job_status=$?

# Clean up the temporary script
rm -f "$temp_script"

# Exit with the job's status
if [ $job_status -ne 0 ]; then
    # Replace %j with actual job ID in error path for display
    display_error_path=${ERROR_PATH//%j/$job_id}
    display_error_path=${display_error_path//%x/$JOB_NAME}
    echo "Job ${job_id} failed. Check ${display_error_path} for details."
    exit 1
fi

echo "Job ${job_id} completed successfully."
exit 0 