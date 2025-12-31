#!/bin/bash

# Script to run different mesh refinement cases for 3D acoustic simulation
# Usage: ./run_cases.sh [start_case] [end_case]
# Example: ./run_cases.sh 1 5  (runs cases 1 through 5)
# Example: ./run_cases.sh       (runs all cases 1 through 13)

# Activate Firedrake environment
source /Users/ddolci/work/venv-firedrake/bin/activate

# Define cases with their mesh sizes (case_number elem_x elem_y elem_z)
# Refined meshes for target error < 0.1
declare -a cases=(
    "1 40 40 20"
    "2 50 50 25"
    "3 60 60 30"
    "4 70 70 35"
    "5 80 80 40"
    "6 90 90 45"
    "7 100 100 50"
    "8 110 110 55"
    "9 120 120 60"
    "10 130 130 65"
    "11 140 140 70"
    "12 150 150 75"
    "13 160 160 80"
    "ref 180 180 90"
)

# Define which mesh types to run (hex, tet, or both)
RUN_HEX=${RUN_HEX:-false}
RUN_TET=${RUN_TET:-true}

# Get start and end case from arguments, default to all cases
START=${1:-1}
END=${2:-13}

# Create output directory if it doesn't exist
mkdir -p comp_performance

# echo "=========================================="
# echo "Running 3D Acoustic Wave Simulation Cases"
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
    # echo "Running Case $case_num: ${elem_x}x${elem_y}x${elem_z} mesh"
    # echo "=========================================="
    # echo "Start time: $(date)"
    
    # Run the simulation in parallel with 4 processes
    # Run both hexahedral and tetrahedral meshes
    
    hex_exit=0
    tet_exit=0
    
    # Hexahedral mesh
    if [ "$RUN_HEX" = true ]; then
        echo "--- Running Hexahedral mesh ---"
        mpiexec -np 4 python3 eage_3d_acoustic.py --case "$case_num" --elem-x "$elem_x" --elem-y "$elem_y" --elem-z "$elem_z"
        hex_exit=$?
    fi
    
    # Tetrahedral mesh
    if [ "$RUN_TET" = true ]; then
        echo "--- Running Tetrahedral mesh ---"
        mpiexec -np 4 python3 eage_3d_acoustic_tet.py --case "$case_num" --elem-x "$elem_x" --elem-y "$elem_y" --elem-z "$elem_z"
        tet_exit=$?
    fi
    
    # Check if simulations succeeded
    if [ "$RUN_HEX" = true ] && [ "$RUN_TET" = true ]; then
        if [ $hex_exit -eq 0 ] && [ $tet_exit -eq 0 ]; then
            echo "✓ Case $case_num completed successfully (hex and tet)"
        elif [ $hex_exit -eq 0 ]; then
            echo "✓ Case $case_num: hex completed, tet failed (exit $tet_exit)"
        elif [ $tet_exit -eq 0 ]; then
            echo "✓ Case $case_num: tet completed, hex failed (exit $hex_exit)"
        else
            echo "✗ Case $case_num failed: hex exit $hex_exit, tet exit $tet_exit"
        fi
    elif [ "$RUN_HEX" = true ]; then
        if [ $hex_exit -eq 0 ]; then
            echo "✓ Case $case_num (hex) completed successfully"
        else
            echo "✗ Case $case_num (hex) failed with exit code $hex_exit"
        fi
    elif [ "$RUN_TET" = true ]; then
        if [ $tet_exit -eq 0 ]; then
            echo "✓ Case $case_num (tet) completed successfully"
        else
            echo "✗ Case $case_num (tet) failed with exit code $tet_exit"
        fi
    fi
    
    # echo "End time: $(date)"
    # echo ""
done

# echo "=========================================="
# echo "All requested cases completed"
# echo "Results saved in comp_performance/"
# echo "=========================================="
