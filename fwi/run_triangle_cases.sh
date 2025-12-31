#!/bin/bash

# Script to run acoustic triangle cases 4 through reference
# Usage: ./run_triangle_cases.sh [start_case] [end_case]

set -e  # Exit on error

# Function to get mesh size for a case
get_mesh_size() {
    local case_num=$1
    case $case_num in
        1) echo "68 34" ;;
        2) echo "102 51" ;;
        3) echo "139 67" ;;
        4) echo "172 85" ;;
        5) echo "205 102" ;;
        6) echo "239 119" ;;
        7) echo "274 136" ;;
        8) echo "290 145" ;;
        9) echo "340 170" ;;
        10) echo "390 180" ;;
        11) echo "418 204" ;;
        12) echo "425 214" ;;
        13) echo "444 222" ;;
        ref|reference) echo "502 251" ;;
        *) echo "" ;;
    esac
}

# Parse arguments
START_CASE=${1:-4}
END_CASE=${2:-13}

echo "=========================================="
echo "Running Acoustic Triangle Cases $START_CASE to $END_CASE"
echo "=========================================="
echo ""

ORIGINAL_FILE="mm_error_acoustic_tringle_no_adapt.py"

# Run cases
for case_num in $(seq $START_CASE $END_CASE); do
    mesh_size=$(get_mesh_size $case_num)
    
    if [ -n "$mesh_size" ]; then
        mesh_values=($mesh_size)
        elem_x=${mesh_values[0]}
        elem_y=${mesh_values[1]}
        
        echo "----------------------------------------"
        echo "Starting case $case_num at $(date)"
        echo "Mesh: ${elem_x}x${elem_y}"
        echo "----------------------------------------"
        
        # Run the simulation with arguments
        if python3 "$ORIGINAL_FILE" --case $case_num --elem-x $elem_x --elem-y $elem_y; then
            echo "✓ Case $case_num completed successfully"
            echo ""
        else
            echo "✗ Case $case_num failed with exit code $?"
            echo "Stopping execution."
            exit 1
        fi
    fi
done

# Run reference case if requested
if [ "$END_CASE" = "ref" ] || [ "$END_CASE" = "reference" ] || ([ "$END_CASE" -ge 14 ] 2>/dev/null); then
    mesh_size=$(get_mesh_size ref)
    mesh_values=($mesh_size)
    elem_x=${mesh_values[0]}
    elem_y=${mesh_values[1]}
    
    echo "----------------------------------------"
    echo "Starting reference case at $(date)"
    echo "Mesh: ${elem_x}x${elem_y}"
    echo "----------------------------------------"
    
    if python3 "$ORIGINAL_FILE" --case ref --elem-x $elem_x --elem-y $elem_y; then
        echo "✓ Reference case completed successfully"
        echo ""
    else
        echo "✗ Reference case failed with exit code $?"
    fi
fi

echo "=========================================="
echo "All cases completed at $(date)"
echo "=========================================="
