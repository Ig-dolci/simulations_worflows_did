#!/bin/bash

# Script to run adaptive triangular mesh cases for acoustic simulation
# Meshes range from mm_acoustic_3.msh to mm_acoustic_75.msh

# Function to get available mesh cases
get_available_cases() {
    echo "3 4 5 6 7 23 27 33 36 45 55 65 75"
}

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_case> <end_case>"
    echo "Available cases: $(get_available_cases)"
    exit 1
fi

START_CASE=$1
END_CASE=$2

# Available cases as array
AVAILABLE_CASES=(3 4 5 6 7 23 27 33 36 45 55 65 75)

echo "=========================================="
echo "Running Adaptive Triangular Mesh Cases"
echo "From case ${START_CASE} to ${END_CASE}"
echo "=========================================="

# Activate virtual environment
source /Users/ddolci/work/venv-firedrake/bin/activate

# Counter for successful runs
SUCCESS_COUNT=0
TOTAL_COUNT=0

# Run each case
for case_num in "${AVAILABLE_CASES[@]}"; do
    # Check if case is within range
    if [ "$case_num" -ge "$START_CASE" ] && [ "$case_num" -le "$END_CASE" ]; then
        echo ""
        echo "=========================================="
        echo "Running Case ${case_num}"
        echo "Mesh: mm_acoustic_${case_num}.msh"
        echo "=========================================="
        
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        
        python mm_error_acoustic_triangle_adaptive.py --case ${case_num}
        
        if [ $? -eq 0 ]; then
            echo "✓ Case ${case_num} completed successfully"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "✗ Case ${case_num} failed"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Completed: ${SUCCESS_COUNT}/${TOTAL_COUNT} cases"
echo "=========================================="
