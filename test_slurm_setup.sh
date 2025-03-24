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

# Create a temporary directory for the container
TMP_DIR=$(mktemp -d)
cd $TMP_DIR

# Copy the Singularity definition file and project files
cp /home/dl1070/projects/cit-sci-traits/cit-sci-traits.def .
cp -r /home/dl1070/projects/cit-sci-traits/* .

# Step 1: Test Singularity container build
echo "Step 1: Testing Singularity container build..."
if singularity build --force cit-sci-traits-test.sif cit-sci-traits.def; then
    print_status "Singularity container built successfully"
else
    print_error "Singularity container build failed"
    cd -
    rm -rf $TMP_DIR
    exit 1
fi

# Step 2: Test requirements in container
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

if singularity run --bind /home/dl1070/projects/cit-sci-traits:/app cit-sci-traits-test.sif python test_imports.py; then
    print_status "Requirements check passed"
else
    print_error "Requirements check failed"
    cd -
    rm -rf $TMP_DIR
    exit 1
fi

# Step 3: Test SLURM task creation
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
#SBATCH --partition=Genoa

# Create a temporary directory for the container
TMP_DIR=$(mktemp -d)
cd $TMP_DIR

# Copy the container and project files
cp /home/dl1070/projects/cit-sci-traits/cit-sci-traits-test.sif .
cp -r /home/dl1070/projects/cit-sci-traits/* .

# Run a simple test
singularity run --bind /home/dl1070/projects/cit-sci-traits:/app cit-sci-traits-test.sif echo "Test SLURM job running"

# Clean up
cd -
rm -rf $TMP_DIR
EOF

if sbatch test_slurm.slurm; then
    print_status "SLURM task creation successful"
else
    print_error "SLURM task creation failed"
    cd -
    rm -rf $TMP_DIR
    exit 1
fi

# Step 4: Test train_models with dry-run
echo -e "\nStep 4: Testing train_models with dry-run..."
if singularity run --bind /home/dl1070/projects/cit-sci-traits:/app cit-sci-traits-test.sif --dry-run splot_gbif; then
    print_status "Dry-run test passed"
else
    print_error "Dry-run test failed"
    cd -
    rm -rf $TMP_DIR
    exit 1
fi

# Step 5: Test train_models with single trait
echo -e "\nStep 5: Testing train_models with single trait..."
if singularity run --bind /home/dl1070/projects/cit-sci-traits:/app cit-sci-traits-test.sif --trait-index 0 splot_gbif; then
    print_status "Single trait test passed"
else
    print_error "Single trait test failed"
    cd -
    rm -rf $TMP_DIR
    exit 1
fi

# Step 6: Test train_models with multiple traits
echo -e "\nStep 6: Testing train_models with multiple traits..."
if singularity run --bind /home/dl1070/projects/cit-sci-traits:/app cit-sci-traits-test.sif --trait-index 0 --trait-index 1 splot_gbif; then
    print_status "Multiple traits test passed"
else
    print_error "Multiple traits test failed"
    cd -
    rm -rf $TMP_DIR
    exit 1
fi

# Clean up
cd -
rm -rf $TMP_DIR

print_status "All tests completed successfully!" 