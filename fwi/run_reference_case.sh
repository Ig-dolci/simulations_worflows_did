#!/bin/bash

# Script to run reference case for quadrilateral mesh only
# Reference case uses 540x270 mesh resolution

echo "=========================================="
echo "Running Reference Case (Quadrilateral)"
echo "=========================================="

# Reference mesh configuration for quadrilateral mesh
REF_QUAD_X=540
REF_QUAD_Y=270

# Run quadrilateral reference case
echo ""
echo "Running Quadrilateral Reference Case..."
echo "Mesh: ${REF_QUAD_X} x ${REF_QUAD_Y}"
source /Users/ddolci/work/venv-firedrake/bin/activate
python mm_error_acoustic_quads.py --case ref --elem-x ${REF_QUAD_X} --elem-y ${REF_QUAD_Y}

if [ $? -eq 0 ]; then
    echo "✓ Quadrilateral reference case completed successfully"
    echo "✓ Output saved to: comp_performance/rec_data_quad_no_adaptive_acoustic_ref.npy"
else
    echo "✗ Quadrilateral reference case failed"
fi

echo ""
echo "=========================================="
echo "Reference case completed!"
echo "=========================================="
