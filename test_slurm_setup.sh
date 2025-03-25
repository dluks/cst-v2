#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

# Function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --all           Run all steps"
    echo "  --step1         Test Singularity container build"
    echo "  --step2         Test requirements in container"
    echo "  --step3         Test SLURM task creation"
    echo "  --step4         Test train_models with dry-run"
    echo "  --step5         Test train_models with single trait"
    echo "  --step6         Test train_models with multiple traits"
    echo "  -h, --help      Show this help message"
    exit 1
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    print_usage
fi

# Initialize flags
RUN_ALL=false
RUN_STEP1=false
RUN_STEP2=false
RUN_STEP3=false
RUN_STEP4=false
RUN_STEP5=false
RUN_STEP6=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --step1)
            RUN_STEP1=true
            shift
            ;;
        --step2)
            RUN_STEP2=true
            shift
            ;;
        --step3)
            RUN_STEP3=true
            shift
            ;;
        --step4)
            RUN_STEP4=true
            shift
            ;;
        --step5)
            RUN_STEP5=true
            shift
            ;;
        --step6)
            RUN_STEP6=true
            shift
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Create a temporary directory for the container
# TMP_DIR=$(mktemp -d)
TMP_DIR=$(pwd)/tmp_$(date +%Y%m%d_%H%M%S)
mkdir -p $TMP_DIR
cd $TMP_DIR
# Copy the Singularity definition file and project files
cp -r $HOME/projects/cit-sci-traits/{pyproject.toml,poetry.lock,README.md,reference,results,src,.env,cit-sci-traits.sif,params.yaml} .

# Function to clean up and exit
cleanup_and_exit() {
    cd -
    rm -rf $TMP_DIR
    exit $1
}

# Step 1: Test Singularity container build
if [ "$RUN_ALL" = true ] || [ "$RUN_STEP1" = true ]; then
    echo "Step 1: Testing Singularity container build..."
    if singularity build --force cit-sci-traits-test.sif cit-sci-traits.def; then
        print_status "Singularity container built successfully"
    else
        print_error "Singularity container build failed"
        cleanup_and_exit 1
    fi
fi

# Step 2: Test requirements in container
if [ "$RUN_ALL" = true ] || [ "$RUN_STEP2" = true ]; then
    echo -e "\nStep 2: Testing requirements in container..."
    # Create a Python script to test imports
    cat > test_imports.py << 'EOF'
def test_imports():
    # Test core dependencies
    import autogluon
    import box
    import numpy
    import pandas
    import scikit_learn
    import torch
    
    # Test project-specific imports
    from src.conf.conf import get_config
    from src.conf.environment import activate_env
    from src.models import autogluon as project_autogluon
    
    print("All core dependencies imported successfully")
    
    # Test project configuration
    cfg = get_config()
    print("Project configuration loaded successfully")
    
    # Test environment activation
    activate_env()
    print("Environment activated successfully")
    
    print("All requirements satisfied")

if __name__ == "__main__":
    test_imports()
EOF

    if singularity run --bind test_imports.py:/app/test_imports.py,src:/app/src,.env:/app/.env cit-sci-traits-test.sif test_imports.py; then
        print_status "Requirements check passed"
    else
        print_error "Requirements check failed"
        cleanup_and_exit 1
    fi
fi

# Step 3: Test SLURM task creation
if [ "$RUN_ALL" = true ] || [ "$RUN_STEP3" = true ]; then
    echo -e "\nStep 3: Testing SLURM task creation..."
    # Create a test SLURM script with minimal resources
    cat > test_slurm.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=test_slurm
#SBATCH --output=test_slurm_%j.log
#SBATCH --error=test_slurm_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

# Create a temporary directory for the container
# TMP_DIR=$(mktemp -d)
# cd $TMP_DIR

# Copy the container and project files
# cp -r $HOME/projects/cit-sci-traits/* .

# Run a simple test
singularity run --bind src:/app/src,.env:/app/.env cit-sci-traits.sif echo "Test SLURM job running"

# Clean up
# cd -
# rm -rf $TMP_DIR
EOF

    if sbatch test_slurm.slurm; then
        print_status "SLURM task creation successful"
    else
        print_error "SLURM task creation failed"
        cleanup_and_exit 1
    fi
fi

# Step 4: Test train_models with dry-run
if [ "$RUN_ALL" = true ] || [ "$RUN_STEP4" = true ]; then
    echo -e "\nStep 4: Testing train_models with dry-run..."
    if singularity run --bind $HOME/projects/cit-sci-traits:/app cit-sci-traits-test.sif --dry-run splot_gbif; then
        print_status "Dry-run test passed"
    else
        print_error "Dry-run test failed"
        cleanup_and_exit 1
    fi
fi

# Step 5: Test train_models with single trait
if [ "$RUN_ALL" = true ] || [ "$RUN_STEP5" = true ]; then
    echo -e "\nStep 5: Testing train_models with single trait..."
    if singularity run --bind $HOME/projects/cit-sci-traits:/app cit-sci-traits-test.sif --trait-index 0 splot_gbif; then
        print_status "Single trait test passed"
    else
        print_error "Single trait test failed"
        cleanup_and_exit 1
    fi
fi

# Step 6: Test train_models with multiple traits
if [ "$RUN_ALL" = true ] || [ "$RUN_STEP6" = true ]; then
    echo -e "\nStep 6: Testing train_models with multiple traits..."
    if singularity run --bind $HOME/projects/cit-sci-traits:/app cit-sci-traits-test.sif --trait-index 0 --trait-index 1 splot_gbif; then
        print_status "Multiple traits test passed"
    else
        print_error "Multiple traits test failed"
        cleanup_and_exit 1
    fi
fi

# Clean up
# cleanup_and_exit 0

print_status "All selected tests completed successfully!" 