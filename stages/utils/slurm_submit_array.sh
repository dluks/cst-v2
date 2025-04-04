#!/bin/bash
# Utility script for submitting multiple jobs to Slurm and waiting for all to complete
# Uses named parameters for better readability

# Default values
JOB_NAME="dvc_array"
OUTPUT_PATH="logs/%x_%A_%a.log"
ERROR_PATH="logs/%x_%A_%a.err"
TIME="01:00:00"
NODES=1
NTASKS=1
CPUS=1
MEM="4G"
PARTITION="cpu"
COMMANDS_FILE=""

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
    --commands-file=*)
      COMMANDS_FILE="${1#*=}"
      shift
      ;;
    *)
      # If no named parameter is given, assume it's the commands file
      if [ -z "$COMMANDS_FILE" ]; then
        COMMANDS_FILE="$1"
        shift
      else
        echo "Unknown parameter: $1"
        exit 1
      fi
      ;;
  esac
done

# Validate commands file
if [ -z "$COMMANDS_FILE" ] || [ ! -f "$COMMANDS_FILE" ]; then
  echo "Error: Commands file not provided or does not exist"
  echo "Usage: $0 --job-name=<name> --output=<path> --error=<path> --time=<time> --nodes=<n> --ntasks=<n> --cpus=<n> --mem=<mem> --partition=<part> --commands-file=<file>"
  echo "Or: $0 [named_params] <commands_file>"
  exit 1
fi

# Count number of tasks
task_count=$(wc -l < "$COMMANDS_FILE")

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
#SBATCH --array=1-$task_count

# Get the command for this array task
command=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")

# Run the command
\$command
EOF

# Make script executable
chmod +x "$temp_script"

# Submit the job
job_id=$(sbatch --parsable "$temp_script")
echo "Submitted job array ${job_id}, waiting for completion..."

# Wait for the job array to complete
srun --dependency=afterany:${job_id} --quiet sleep 1

# Check all job statuses
failed_jobs=$(sacct -j ${job_id} --format=JobID,State --noheader | grep -v "COMPLETED" | grep -v "PENDING" | grep -v "RUNNING" | wc -l)

# Clean up the temporary script
rm -f "$temp_script"

# Exit with appropriate status
if [ $failed_jobs -gt 0 ]; then
    echo "$failed_jobs tasks in job array ${job_id} failed. Check logs for details."
    exit 1
fi

echo "All tasks in job array ${job_id} completed successfully."
exit 0 