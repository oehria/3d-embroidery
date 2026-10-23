#!/bin/bash
res_folder=$1
output_folder=$2

mkdir -p $2

python scripts/sample_stitches.py $res_folder/pattern.svg -o $2/sampled.obj --interval 0.5

python scripts/prepare_stitch_mesh.py $2/sampled.obj $2/shell.obj --max-area 0.002 --boundary $2/sampled_boundary.obj

python scripts/lift_polylines_to_mesh.py $2/sampled.obj $1/shrinky_out.obj -o $2/lifted.obj