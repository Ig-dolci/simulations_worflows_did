#!/bin/bash

# Script to run different mesh refinement cases for 3D acoustic simulation
# Usage: ./run_3d_acoustic_tet_cases.sh [start_case] [end_case]
# Example: ./run_3d_acoustic_tet_cases.sh 1 5  (runs cases 1 through 5)
# Example: ./run_3d_acoustic_tet_cases.sh       (runs all cases 1 through 13)

# Activate Firedrake environment
source /Users/ddolci/work/venv-firedrake/bin/activate

# Define cases with their mesh sizes (case_number elem_x elem_y elem_z)
# Tetrahedral meshes use KMV elements
declare -a cases=(
    "1 28 28 14"
    "2 36 36 18"
    "3 43 43 21"
    "4 51 51 25"
    "5 57 57 28"
    "6 64 64 32"
    "7 71 71 35"
    "8 79 79 39"
    "9 85 85 42"
    "10 93 93 46"
    "11 99 99 49"
    "12 107 107 53"
    "13 113 113 56"
    "ref 129 129 64"
)

# Get start and end case from arguments, default to all cases
START=${1:-1}
END=${2:-13}

# Create output directory if it doesn't exist
mkdir -p comp_performance

# echo "=========================================="
# echo "Running 3D Acoustic Wave Simulation Cases (Tetrahedral Elements)"
# echo "=========================================="
# echo "Start case: $START"
# echo "End case: $END"
# echo ""

# Run each case
for case_info in "${cases[@]}"; do
    read -r case_num elem_x elem_y elem_z <<< "$case_info"
    
    # Skip if outside requested range (skip ref case for numeric comparison)
    if [[ "$case_num" =~ ^[0-9]+$ ]]; then
        if [ "$case_num" -lt "$START" ] || [ "$case_num" -gt "$END" ]; then
            continue
        fi
    fi
    
    # echo "=========================================="
    # echo "Running Case $case_num: ${elem_x}x${elem_y}x${elem_z} mesh (Tetrahedral)"
    # echo "=========================================="
    # echo "Start time: $(date)"
    
    # Run the simulation in parallel with 4 processes
    echo "--- Running Tetrahedral mesh (${elem_x}x${elem_y}x${elem_z}) ---"
    mpiexec -np 4 python3 eage_3d_acoustic_tet.py --case "$case_num" --elem-x "$elem_x" --elem-y "$elem_y" --elem-z "$elem_z"
    tet_exit=$?
    
    # Check if simulation succeeded
    if [ $tet_exit -eq 0 ]; then
        echo "✓ Case $case_num (tet) completed successfully"
    else
        echo "✗ Case $case_num (tet) failed with exit code $tet_exit"
    fi
    
    # echo "End time: $(date)"
    # echo ""
done

# echo "=========================================="
# echo "All requested cases completed"
# echo "Results saved in comp_performance/"
# echo "=========================================="
