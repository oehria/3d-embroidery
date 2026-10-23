#!/usr/bin/env python3
"""
Lift 2D polylines onto a 3D mesh via UV parameterization.

Given:
  1. An OBJ file with 2D polylines (vertices with y=0, edges via 'l' lines)
  2. An OBJ file with a 3D mesh that has UV coordinates (vt entries, f v/vt/... faces)

For each polyline vertex, we find the UV triangle it lies in, compute barycentric
coordinates, and interpolate the corresponding 3D position on the mesh surface.

Usage:
  python lift_polylines_to_mesh.py polylines.obj mesh.obj -o output.obj
"""

import argparse
import numpy as np
from collections import defaultdict


def parse_polyline_obj(filepath):
    """Parse an OBJ with 2D polyline vertices and 'l' (line) edges."""
    vertices = []
    edges = []  # list of (i, j) 0-indexed

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                x, y = float(parts[1]), -float(parts[3])
                vertices.append((x, y))
            elif parts[0] == "l":
                # OBJ indices are 1-based
                idx = [int(p) - 1 for p in parts[1:]]
                for k in range(len(idx) - 1):
                    edges.append((idx[k], idx[k + 1]))

    return np.array(vertices, dtype=np.float64), edges


def parse_mesh_obj(filepath):
    """Parse an OBJ mesh with 3D vertices, UV coords, and faces.

    Returns:
        positions: (N, 3) array of 3D vertex positions
        uvs: (M, 2) array of UV coordinates
        faces_pos: list of face tuples (0-indexed position indices)
        faces_uv: list of face tuples (0-indexed UV indices)
    """
    positions = []
    uvs = []
    faces_pos = []
    faces_uv = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            if parts[0] == "v" and len(parts) >= 4:
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt":
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f":
                fv = []
                ft = []
                for token in parts[1:]:
                    components = token.split("/")
                    fv.append(int(components[0]) - 1)
                    if len(components) >= 2 and components[1]:
                        ft.append(int(components[1]) - 1)
                faces_pos.append(tuple(fv))
                if ft:
                    faces_uv.append(tuple(ft))

    return (
        np.array(positions, dtype=np.float64),
        np.array(uvs, dtype=np.float64),
        faces_pos,
        faces_uv,
    )


def barycentric_coords_2d(p, a, b, c):
    """Compute barycentric coordinates of point p w.r.t. triangle (a, b, c) in 2D.

    Returns (u, v, w) such that p = u*a + v*b + w*c.
    Returns None if the triangle is degenerate.
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-30:
        return None

    inv_denom = 1.0 / denom
    w = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    u = 1.0 - v - w

    return u, v, w


def build_uv_grid(uvs, faces_uv, grid_res=64):
    """Build a spatial hash grid over UV triangles for fast lookup."""
    # Compute bounding box of all UVs
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    # Add small padding
    uv_min -= uv_range * 0.01
    uv_max += uv_range * 0.01
    uv_range = uv_max - uv_min

    grid = defaultdict(list)
    cell_size = uv_range / grid_res

    for fi, face_uv in enumerate(faces_uv):
        # Get UV coords for this face (triangulate if needed)
        tri_uvs = uvs[list(face_uv)]
        tri_min = tri_uvs.min(axis=0)
        tri_max = tri_uvs.max(axis=0)

        # Grid cells this triangle overlaps
        ci_min = np.clip(((tri_min - uv_min) / cell_size).astype(int), 0, grid_res - 1)
        ci_max = np.clip(((tri_max - uv_min) / cell_size).astype(int), 0, grid_res - 1)

        for ci in range(ci_min[0], ci_max[0] + 1):
            for cj in range(ci_min[1], ci_max[1] + 1):
                grid[(ci, cj)].append(fi)

    return grid, uv_min, cell_size, grid_res


def triangulate_face(face):
    """Fan-triangulate a polygon face into triangles. Returns list of 3-tuples."""
    tris = []
    for i in range(1, len(face) - 1):
        tris.append((face[0], face[i], face[i + 1]))
    return tris


def find_uv_triangle(
    point, uvs, faces_uv, faces_pos, grid, uv_min, cell_size, grid_res, tolerance=1e-6
):
    """Find which UV triangle contains the 2D point, return barycentric coords and face info.

    Returns:
        (bary_coords, face_pos_indices) or (None, None) if not found
    """
    ci = int((point[0] - uv_min[0]) / cell_size[0])
    cj = int((point[1] - uv_min[1]) / cell_size[1])
    ci = np.clip(ci, 0, grid_res - 1)
    cj = np.clip(cj, 0, grid_res - 1)

    # Search in neighboring cells too for robustness
    best_result = None
    best_min_bary = -np.inf

    for di in range(-1, 2):
        for dj in range(-1, 2):
            key = (ci + di, cj + dj)
            if key not in grid:
                continue
            for fi in grid[key]:
                face_uv = faces_uv[fi]
                face_pos = faces_pos[fi]

                # Triangulate if face has more than 3 vertices
                uv_tris = triangulate_face(face_uv)
                pos_tris = triangulate_face(face_pos)

                for uv_tri, pos_tri in zip(uv_tris, pos_tris):
                    a = uvs[uv_tri[0]]
                    b = uvs[uv_tri[1]]
                    c = uvs[uv_tri[2]]

                    bary = barycentric_coords_2d(point, a, b, c)
                    if bary is None:
                        continue

                    u, v, w = bary
                    min_bary = min(u, v, w)

                    if min_bary >= -tolerance:
                        # Point is inside (or on boundary of) this triangle
                        return bary, pos_tri

                    # Track closest triangle for fallback
                    if best_result is None or min_bary > best_min_bary:
                        best_min_bary = min_bary
                        best_result = (bary, pos_tri)

    return best_result if best_result else (None, None)


def lift_point(
    point, positions, uvs, faces_uv, faces_pos, grid, uv_min, cell_size, grid_res
):
    """Lift a 2D point to 3D using the UV parameterization."""
    bary, pos_tri = find_uv_triangle(
        point, uvs, faces_uv, faces_pos, grid, uv_min, cell_size, grid_res
    )

    if bary is None:
        return None

    u, v, w = bary
    p3d = (
        u * positions[pos_tri[0]]
        + v * positions[pos_tri[1]]
        + w * positions[pos_tri[2]]
    )
    return p3d


def brute_force_find(point, uvs, faces_uv, faces_pos, positions, tolerance=1e-5):
    """Brute-force fallback: search all UV triangles."""
    best_result = None
    best_min_bary = -np.inf

    for fi in range(len(faces_uv)):
        face_uv = faces_uv[fi]
        face_pos = faces_pos[fi]

        uv_tris = triangulate_face(face_uv)
        pos_tris = triangulate_face(face_pos)

        for uv_tri, pos_tri in zip(uv_tris, pos_tris):
            a = uvs[uv_tri[0]]
            b = uvs[uv_tri[1]]
            c = uvs[uv_tri[2]]

            bary = barycentric_coords_2d(point, a, b, c)
            if bary is None:
                continue

            u, v, w = bary
            min_bary = min(u, v, w)

            if min_bary >= -tolerance:
                p3d = (
                    u * positions[pos_tri[0]]
                    + v * positions[pos_tri[1]]
                    + w * positions[pos_tri[2]]
                )
                return p3d

            if best_result is None or min_bary > best_min_bary:
                best_min_bary = min_bary
                u_c = max(0, u)
                v_c = max(0, v)
                w_c = max(0, w)
                s = u_c + v_c + w_c
                if s > 0:
                    u_c /= s
                    v_c /= s
                    w_c /= s
                p3d = (
                    u_c * positions[pos_tri[0]]
                    + v_c * positions[pos_tri[1]]
                    + w_c * positions[pos_tri[2]]
                )
                best_result = p3d
                best_min_bary = min_bary

    return best_result


def write_obj(filepath, vertices_3d, edges):
    """Write lifted polylines as an OBJ file."""
    with open(filepath, "w") as f:
        f.write(
            f"# Lifted polylines ({len(vertices_3d)} vertices, {len(edges)} edges)\n"
        )
        for v in vertices_3d:
            f.write(f"v {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n")
        for i, j in edges:
            f.write(f"l {i + 1} {j + 1}\n")


def print_edge_distances(poly_verts, lifted_verts, poly_edges):
    """Print original (2D) and lifted (3D) edge lengths, and the stretch percentage."""
    print("\n--- Edge Distance Report ---")
    print(f"{'Edge':<12} {'Orig 2D':>12} {'Lifted 3D':>12} {'Stretch %':>12}")
    print("-" * 50)

    stretches = []
    for i, (a, b) in enumerate(poly_edges):
        # Original 2D distance
        d_orig = np.linalg.norm(poly_verts[a] - poly_verts[b])
        # Lifted 3D distance
        d_lift = np.linalg.norm(lifted_verts[a] - lifted_verts[b])
        # Stretch percentage
        if d_orig > 0:
            stretch = (d_lift / d_orig - 1.0) * 100.0
        else:
            stretch = 0.0
        stretches.append(stretch)
        # print(f"({a:>4},{b:>4})  {d_orig:>12.6f} {d_lift:>12.6f} {stretch:>11.2f}%")

    if stretches:
        print("-" * 50)
        print(f"{'Min stretch:':<26} {min(stretches):>11.2f}%")
        print(f"{'Max stretch:':<26} {max(stretches):>11.2f}%")
        print(f"{'Avg stretch:':<26} {np.mean(stretches):>11.2f}%")
    print()


def print_distance_violations(poly_verts, lifted_verts):
    """Check all vertex pairs for distance increase in the lifted configuration.

    Reports pairs where the lifted 3D distance exceeds the original 2D distance.
    """
    n = len(poly_verts)
    total_pairs = n * (n - 1) // 2
    print(f"\n--- Distance Violation Report (all {total_pairs} vertex pairs) ---")

    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            d_orig = np.linalg.norm(poly_verts[i] - poly_verts[j])
            d_lift = np.linalg.norm(lifted_verts[i] - lifted_verts[j])
            if d_lift > d_orig and d_orig > 0:
                increase = (d_lift / d_orig - 1.0) * 100.0
                violations.append((i, j, d_orig, d_lift, increase))

    # if violations:
    #     print(f"{'Pair':<12} {'Orig 2D':>12} {'Lifted 3D':>12} {'Increase %':>12}")
    #     print("-" * 50)
    #     for i, j, d_orig, d_lift, increase in violations:
    #         print(
    #             f"({i:>4},{j:>4})  {d_orig:>12.6f} {d_lift:>12.6f} {increase:>11.2f}%"
    #         )
    #     print("-" * 50)

    print(
        f"Violations: {len(violations)} / {total_pairs} pairs "
        f"({len(violations) / total_pairs * 100:.2f}% of all pairs)"
    )
    if violations:
        increases = [v[4] for v in violations]
        print(f"  Max increase: {max(increases):.2f}%")
        print(f"  Avg increase: {np.mean(increases):.2f}%")
    else:
        print("No distance increases detected.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Lift 2D polylines onto a 3D mesh via UV parameterization."
    )
    parser.add_argument(
        "polylines", help="OBJ file with 2D polylines (vertices + 'l' edges)"
    )
    parser.add_argument("mesh", help="OBJ file with 3D mesh and UV coordinates")
    parser.add_argument(
        "-o",
        "--output",
        default="lifted_polylines.obj",
        help="Output OBJ file (default: lifted_polylines.obj)",
    )
    parser.add_argument(
        "--grid-res",
        type=int,
        default=64,
        help="Spatial hash grid resolution (default: 64)",
    )
    args = parser.parse_args()

    # Parse inputs
    print(f"Reading polylines from: {args.polylines}")
    poly_verts, poly_edges = parse_polyline_obj(args.polylines)
    print(f"  {len(poly_verts)} vertices, {len(poly_edges)} edges")

    print(f"Reading mesh from: {args.mesh}")
    positions, uvs, faces_pos, faces_uv = parse_mesh_obj(args.mesh)
    print(f"  {len(positions)} vertices, {len(uvs)} UVs, {len(faces_pos)} faces")

    if len(faces_uv) == 0:
        print("ERROR: Mesh has no UV coordinates. Cannot perform UV-based lifting.")
        return

    if len(faces_uv) != len(faces_pos):
        print(
            f"WARNING: {len(faces_pos)} position faces but {len(faces_uv)} UV faces. "
            f"Using min({len(faces_pos)}, {len(faces_uv)}) faces."
        )
        n = min(len(faces_pos), len(faces_uv))
        faces_pos = faces_pos[:n]
        faces_uv = faces_uv[:n]

    # Build spatial index
    print("Building UV spatial index...")
    grid, uv_min, cell_size, grid_res = build_uv_grid(uvs, faces_uv, args.grid_res)

    # Lift each polyline vertex
    print("Lifting polyline vertices...")
    lifted_verts = []
    failed = 0
    for i, pt in enumerate(poly_verts):
        p3d = lift_point(
            pt, positions, uvs, faces_uv, faces_pos, grid, uv_min, cell_size, grid_res
        )

        if p3d is None:
            # Brute-force fallback
            p3d = brute_force_find(pt, uvs, faces_uv, faces_pos, positions)

        if p3d is not None:
            lifted_verts.append(p3d)
        else:
            print(
                f"  WARNING: vertex {i} at ({pt[0]:.6f}, {pt[1]:.6f}) could not be mapped. "
                f"Using (0, 0, 0)."
            )
            lifted_verts.append(np.array([0.0, 0.0, 0.0]))
            failed += 1

    if failed > 0:
        print(
            f"  {failed}/{len(poly_verts)} vertices could not be mapped to any UV triangle."
        )
    else:
        print(f"  All {len(poly_verts)} vertices successfully mapped.")

    # Print edge distance report
    lifted_verts_arr = np.array(lifted_verts)
    # print_edge_distances(poly_verts, lifted_verts_arr, poly_edges)
    # print_distance_violations(poly_verts, lifted_verts_arr)

    # Write output
    print(f"Writing output to: {args.output}")
    write_obj(args.output, lifted_verts, poly_edges)
    print("Done!")


if __name__ == "__main__":
    main()
