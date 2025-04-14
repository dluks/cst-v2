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
NTASKS_PER_NODE=""  # New parameter for tasks per node
CPUS=1
THREADS_PER_CORE=1  # New parameter for threads per core
MEM="500M"
PARTITION="cpu"
CONSTRAINT=""       # New parameter for node constraints
QOS=""              # New parameter for quality of service
ACCOUNT=""          # New parameter for account
MAIL_TYPE="BEGIN,END,FAIL"        # Email notification types
MAIL_USER="daniel.lusk@geosense.uni-freiburg.de"        # Email address for notifications
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
    --ntasks-per-node=*)
      NTASKS_PER_NODE="${1#*=}"
      shift
      ;;
    --cpus=*|--cpus-per-task=*)
      CPUS="${1#*=}"
      shift
      ;;
    --threads-per-core=*)
      THREADS_PER_CORE="${1#*=}"
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
    --constraint=*)
      CONSTRAINT="${1#*=}"
      shift
      ;;
    --qos=*)
      QOS="${1#*=}"
      shift
      ;;
    --account=*)
      ACCOUNT="${1#*=}"
      shift
      ;;
    --mail-type=*)
      MAIL_TYPE="${1#*=}"
      shift
      ;;
    --mail-user=*)
      MAIL_USER="${1#*=}"
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
  echo "Usage: $0 [options] <command>"
  echo "Options:"
  echo "  --job-name=<name>           Job name (default: dvc_job)"
  echo "  --output=<path>             Output file path (default: logs/%x_%j.log)"
  echo "  --error=<path>              Error file path (default: logs/%x_%j.err)"
  echo "  --time=<time>               Maximum runtime (default: 01:00:00)"
  echo "  --nodes=<n>                 Number of nodes (default: 1)"
  echo "  --ntasks=<n>                Total number of tasks (default: 1)"
  echo "  --ntasks-per-node=<n>       Tasks per node (optional)"
  echo "  --cpus=<n>                  CPUs per task (default: 1)"
  echo "  --cpus-per-task=<n>         Alias for --cpus"
  echo "  --threads-per-core=<n>      Threads per core (optional)"
  echo "  --mem=<mem>                 Memory per node (default: 500M)"
  echo "  --partition=<part>          Partition/queue (default: cpu)"
  echo "  --constraint=<features>     Node features/constraints (optional)"
  echo "  --qos=<qos>                 Quality of service (optional)"
  echo "  --account=<account>         Account to charge (optional)"
  echo "  --mail-type=<type>          When to send email: BEGIN,END,FAIL,ALL (optional)"
  echo "  --mail-user=<email>         Email address for notifications (optional)"
  echo "  --command=<cmd>             Command to run (optional, can be last argument)"
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
EOF

# Add optional parameters if specified
[ -n "$NTASKS_PER_NODE" ] && echo "#SBATCH --ntasks-per-node=$NTASKS_PER_NODE" >> "$temp_script"
echo "#SBATCH --cpus-per-task=$CPUS" >> "$temp_script"
[ -n "$THREADS_PER_CORE" ] && echo "#SBATCH --threads-per-core=$THREADS_PER_CORE" >> "$temp_script"
echo "#SBATCH --mem=$MEM" >> "$temp_script"
echo "#SBATCH --partition=$PARTITION" >> "$temp_script"
[ -n "$CONSTRAINT" ] && echo "#SBATCH --constraint=$CONSTRAINT" >> "$temp_script"
[ -n "$QOS" ] && echo "#SBATCH --qos=$QOS" >> "$temp_script"
[ -n "$ACCOUNT" ] && echo "#SBATCH --account=$ACCOUNT" >> "$temp_script"
[ -n "$MAIL_TYPE" ] && echo "#SBATCH --mail-type=$MAIL_TYPE" >> "$temp_script"
[ -n "$MAIL_USER" ] && echo "#SBATCH --mail-user=$MAIL_USER" >> "$temp_script"

# Add job start notification
cat >> "$temp_script" << 'EOF'

# Notify when job starts execution (resources allocated)
job_start_notification() {
    job_id=$SLURM_JOB_ID
    hostname=$(hostname)
    start_time=$(date)
    echo "Job $job_id has started on $hostname at $start_time" > "$SLURM_SUBMIT_DIR/job_${job_id}_start.txt"
    
    # Print allocation info to log
    echo "====== JOB RESOURCE ALLOCATION ======" 
    echo "Job ID: $SLURM_JOB_ID"
    echo "Allocated nodes: $(scontrol show hostnames | tr '\n' ' ')"
    echo "Allocated CPUs: $SLURM_CPUS_ON_NODE total, $SLURM_CPUS_PER_TASK per task"
    echo "Tasks: $SLURM_NTASKS total, $SLURM_NTASKS_PER_NODE per node"
    echo "Start time: $start_time"
    echo "======================================"
}

# Call the notification function
job_start_notification

EOF

# Add the command
cat >> "$temp_script" << EOF

# Run the command
$COMMAND
EOF

# Make script executable
chmod +x "$temp_script"

# Submit the job
job_id=$(sbatch --parsable "$temp_script")
echo "Submitted job ${job_id}, waiting for completion..."

# Replace %j and %x with actual job ID and name in error and output paths
actual_error_path=${ERROR_PATH//%j/$job_id}
actual_error_path=${actual_error_path//%x/$JOB_NAME}

echo "Monitoring job ${job_id}, waiting for completion..."

# Function to check if job is still running
job_is_running() {
    # Add timeout to squeue command to prevent hanging
    timeout 10 squeue -j ${job_id} --noheader &> /dev/null
    local status=$?
    # If timeout or command failed, retry once more before giving up
    if [ $status -ne 0 ]; then
        sleep 2
        timeout 10 squeue -j ${job_id} --noheader &> /dev/null
        status=$?
        # If still failing after retry, check with sacct
        if [ $status -ne 0 ]; then
            timeout 10 sacct -j ${job_id} --format=State --noheader | grep -q "RUNNING\|PENDING"
            status=$?
        fi
    fi
    return $status
}

# Function to kill any tail processes
kill_tail_processes() {
    # Find and kill all tail processes related to our file
    for pid in $(pgrep -f "tail -f ${actual_error_path}" 2>/dev/null); do
        kill -9 $pid 2>/dev/null || true
    done
    # Also kill our specific tail process if we have its PID
    if [ -n "$1" ] && [ "$1" -gt 0 ]; then
        kill -9 $1 2>/dev/null || true
    fi
}

# Function to cancel the Slurm job
cancel_slurm_job() {
    echo "Cancelling Slurm job ${job_id}..."
    scancel ${job_id} 2>/dev/null || true
}

# Register cleanup handler that will run on script exit
cleanup() {
    local exit_code=$?
    echo "Cleaning up monitoring processes..."
    kill_tail_processes
    
    # Only cancel the job if we're being interrupted
    if [ $exit_code -eq 130 ] || [ $exit_code -eq 143 ]; then  # SIGINT (Ctrl-C) or SIGTERM
        cancel_slurm_job
    fi
    
    # Kill any other processes we might have started
    jobs -p | xargs -r kill -9 2>/dev/null || true
    # Clean up the temporary script
    rm -f "$temp_script"
    
    # Only exit with error if we were interrupted
    if [ $exit_code -eq 130 ] || [ $exit_code -eq 143 ]; then
        exit 1
    fi
}
trap cleanup EXIT INT TERM

# Wait for the job to start and create the log file (with timeout)
max_wait=60
wait_count=0
echo "Waiting for log file: ${actual_error_path}"
while [ ! -f "${actual_error_path}" ] && job_is_running; do
    sleep 1
    wait_count=$((wait_count + 1))
    if [ $wait_count -ge $max_wait ]; then
        echo "Timeout waiting for log file. Continuing without tailing..."
        break
    fi
done

# Start tailing in a controlled way
if [ -f "${actual_error_path}" ]; then
    echo "Tailing log file: ${actual_error_path}"
    echo "-------------------------"
    
    # Use a separate subshell with timeout to limit tail duration
    (
        # Start tail in the background
        tail -f "${actual_error_path}" &
        tail_pid=$!
        
        # Check job status every 5 seconds
        while job_is_running; do
            sleep 5
        done
        
        # Job is done - give a few seconds to catch final output
        sleep 3
        
        # Kill the tail process
        kill -9 $tail_pid 2>/dev/null || true
        exit 0
    ) &
    tail_monitor_pid=$!
    
    # Set a maximum wait time (5 minutes) for the tail monitor
    timeout_seconds=300
    wait_seconds=0
    
    # Wait for either the job to finish or timeout
    while [ $wait_seconds -lt $timeout_seconds ] && kill -0 $tail_monitor_pid 2>/dev/null; do
        if ! job_is_running && [ $wait_seconds -gt 10 ]; then
            # If job is done and we've waited at least 10 seconds, break the wait
            sleep 3  # Give a few more seconds for final output
            break
        fi
        sleep 5
        wait_seconds=$((wait_seconds + 5))
    done
    
    # If we're still tailing after timeout, force kill it
    if kill -0 $tail_monitor_pid 2>/dev/null; then
        kill -9 $tail_monitor_pid 2>/dev/null || true
    fi
    
    # Make absolutely sure all tail processes are gone
    kill_tail_processes
    echo "-------------------------"
    echo "Log tailing complete."
else
    echo "Log file not created, waiting for job to complete..."
    # Wait for job completion with robust error handling
    while true; do
        if job_is_running; then
            sleep 5  # Check status every 5 seconds
        else
            echo "Job ${job_id} appears to have completed."
            break
        fi
    done
fi

# Final check and cleanup
echo "Job ${job_id} has finished execution."
# Make sure no tail processes remain
kill_tail_processes
# Reset the trap as we're doing explicit cleanup
trap - EXIT INT TERM

# Check job status with timeout
echo "Checking job status..."
job_status=1  # Default to failure
timeout 15 sacct -j ${job_id} --format=State --noheader > /tmp/job_status_$$ 2>/dev/null
if [ $? -eq 0 ]; then
    grep -q "COMPLETED" /tmp/job_status_$$
    job_status=$?
    rm -f /tmp/job_status_$$
else
    echo "Warning: Could not check job status with sacct, assuming job completed."
    job_status=0  # Assume success if sacct times out
fi

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
if [ -f "$SLURM_SUBMIT_DIR/job_${job_id}_start.txt" ]; then
    echo "Job start notification can be found in: job_${job_id}_start.txt"
fi
exit 0 