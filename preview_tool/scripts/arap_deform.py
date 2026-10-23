#!/usr/bin/env python3
"""
Run ARAP (As-Rigid-As-Possible) deformation using libigl.

Given:
  1. An OBJ file with the input mesh
  2. An OBJ file with positional constraints (target positions)

The first N vertices of the mesh are constrained to match the N vertices
from the constraints file.

Usage:
  python arap_deform.py mesh.obj constraints.obj -o result.obj
  python arap_deform.py mesh.obj constraints.obj -o result.obj --max-iter 20 --num-solves 5
"""

import argparse
import numpy as np
import igl


def main():
    parser = argparse.ArgumentParser(description="Run ARAP deformation using libigl.")
    parser.add_argument(
        "mesh",
        help="Input mesh OBJ file",
    )
    parser.add_argument(
        "constraints",
        help="OBJ file with target vertex positions (N vertices → constrain first N mesh vertices)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="arap_result.obj",
        help="Output OBJ file (default: arap_result.obj)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Max local-global iterations per solve (default: 10)",
    )
    parser.add_argument(
        "--num-solves",
        type=int,
        default=1,
        help="Number of successive ARAP solves to run (default: 1)",
    )
    parser.add_argument(
        "--energy",
        choices=["spokes", "spokes_and_rims", "elements"],
        default="spokes_and_rims",
        help="ARAP energy type (default: spokes_and_rims)",
    )
    args = parser.parse_args()

    energy_map = {
        "spokes": igl.ARAP_ENERGY_TYPE_SPOKES,
        "spokes_and_rims": igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS,
        "elements": igl.ARAP_ENERGY_TYPE_ELEMENTS,
    }

    # ── Read input mesh ──────────────────────────────────────────────────
    print(f"Reading mesh: {args.mesh}")
    V, TC, N, F, FTC, FN = igl.readOBJ(args.mesh)
    print(f"  {V.shape[0]} vertices, {F.shape[0]} faces")

    # ── Read constraint positions ────────────────────────────────────────
    print(f"Reading constraints: {args.constraints}")
    Vc, _, _, Fc, _, _ = igl.readOBJ(args.constraints)
    num_constraints = Vc.shape[0]
    print(f"  {num_constraints} constraint vertices")

    if num_constraints > V.shape[0]:
        print(
            f"ERROR: More constraint vertices ({num_constraints}) than mesh "
            f"vertices ({V.shape[0]})."
        )
        return

    # Constrained vertex indices: first N vertices of the mesh
    b = np.arange(num_constraints, dtype=np.int32)  # boundary condition indices
    bc = Vc[:, : V.shape[1]]  # target positions (match dimension of mesh)

    print(f"  Constraining vertices 0..{num_constraints - 1}")

    # ── ARAP precomputation ──────────────────────────────────────────────
    print("Running ARAP precomputation...")
    arap_data = igl.ARAPData()
    arap_data.max_iter = args.max_iter
    arap_data.energy = energy_map[args.energy]

    dim = V.shape[1]  # 2 or 3
    print("V shape:", V.shape)
    igl.arap_precomputation(V, F, dim, b, arap_data)

    # ── ARAP solve ───────────────────────────────────────────────────────
    U = V.copy()
    for i in range(args.num_solves):
        print(f"ARAP solve {i + 1}/{args.num_solves}...")
        U = igl.arap_solve(bc, arap_data, U)

    # ── Report displacement statistics ───────────────────────────────────
    displacements = np.linalg.norm(U - V, axis=1)
    print(f"\nDisplacement stats:")
    print(f"  Max: {displacements.max():.6f}")
    print(f"  Mean: {displacements.mean():.6f}")
    print(f"  Constrained max: {displacements[:num_constraints].max():.6f}")
    print(f"  Free max: {displacements[num_constraints:].max():.6f}")

    # ── Export result ────────────────────────────────────────────────────
    print(f"\nWriting result to: {args.output}")
    igl.writeOBJ(args.output, U, F)
    print("Done!")


if __name__ == "__main__":
    main()
