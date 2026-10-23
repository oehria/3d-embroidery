#!/usr/bin/env python3
"""
Prepare a stitch mesh for simulation.

Takes an .obj file with 2D vertex locations (y=0, points in XZ plane),
performs Delaunay triangulation with a square boundary, and extrudes the
result in the y direction to produce a thin 3D shell.

Optionally accepts a boundary .obj file (containing vertices and 'l' edges)
to use as the domain boundary instead of the auto-generated bounding box.

Requirements:
    pip install triangle numpy

Usage:
    python prepare_stitch_mesh.py input.obj output.obj [options]

Examples:
    python prepare_stitch_mesh.py pattern.obj shell.obj --max-area 0.001
    python prepare_stitch_mesh.py pattern.obj shell.obj --padding 0.2
    python prepare_stitch_mesh.py pattern.obj shell.obj --boundary boundary.obj
"""

import argparse
import heapq
import sys

import numpy as np

try:
    import triangle as tr
except ImportError:
    print("Error: 'triangle' library is required. Install it with:")
    print("  pip install triangle")
    sys.exit(1)


def read_obj_vertices(filepath):
    """Read vertices and lines from an .obj file.

    Returns:
        vertices: Nx3 array of vertex positions.
        lines: list of (v0, v1) tuples with 1-indexed vertex indices.
    """
    vertices = []
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif len(parts) >= 3 and parts[0] == "l":
                lines.append((int(parts[1]), int(parts[2])))
    if not vertices:
        print(f"Error: No vertices found in {filepath}")
        sys.exit(1)
    return np.array(vertices), lines


def read_boundary_obj(filepath):
    """Read a boundary .obj file containing vertices and line edges.

    The boundary file defines the domain boundary as a polyline/polygon
    via 'v' (vertex) and 'l' (line edge) entries.

    Returns:
        vertices_2d: Nx2 array of (x, z) boundary vertex positions.
        segments: list of (i, j) tuples with 0-indexed local vertex indices.
    """
    vertices = []
    segments = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "v":
                # Store as (x, z), dropping y
                vertices.append([float(parts[1]), float(parts[3])])
            elif len(parts) >= 3 and parts[0] == "l":
                # OBJ is 1-indexed → convert to 0-indexed
                segments.append((int(parts[1]) - 1, int(parts[2]) - 1))
    if not vertices:
        print(f"Error: No vertices found in boundary file {filepath}")
        sys.exit(1)
    if not segments:
        print(f"Error: No line edges found in boundary file {filepath}")
        sys.exit(1)
    return np.array(vertices), segments


def write_obj(filepath, vertices, faces):
    """Write vertices and triangular faces to an .obj file (1-indexed)."""
    with open(filepath, "w") as f:
        f.write(f"# Stitch mesh: {len(vertices)} vertices, {len(faces)} faces\n")
        for v in vertices:
            f.write(f"v {v[0]:.8g} {v[1]:.8g} {v[2]:.8g}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def write_stitch_info(filepath, obj_lines):
    """Write stitch info file from OBJ line elements.

    Each OBJ line 'l x y' (1-indexed) is converted to:
        stitch x-1 y-1 0.3
    (0-indexed vertex indices)
    """
    with open(filepath, "w") as f:
        for v0, v1 in obj_lines:
            f.write(f"stitch {v0 - 1} {v1 - 1} 0.3\n")
    print(f"  Wrote {len(obj_lines)} stitch constraints to {filepath}")


def build_2d_triangulation(points_2d, n_original, max_area, padding_fraction):
    """
    Build a Delaunay triangulation of the 2D points within an extended
    bounding-box boundary.

    Args:
        points_2d: Nx2 array of (x, z) coordinates from the input .obj.
        n_original: Number of original vertices to preserve.
        max_area: Maximum triangle area constraint for mesh refinement.
        padding_fraction: Fraction of bounding box size to add as padding.

    Returns:
        mesh_vertices_2d: Mx2 array of all mesh vertices (original first).
        mesh_triangles: Kx3 array of triangle vertex indices.
    """
    # Compute bounding box
    min_coords = points_2d.min(axis=0)
    max_coords = points_2d.max(axis=0)
    bbox_size = max_coords - min_coords
    padding = padding_fraction * max(bbox_size[0], bbox_size[1])

    # Ensure minimum padding for degenerate cases (e.g. single point / line)
    if padding < 1e-10:
        padding = 1.0

    x_min = min_coords[0] - padding
    x_max = max_coords[0] + padding
    z_min = min_coords[1] - padding
    z_max = max_coords[1] + padding

    print(
        f"  Bounding box: x=[{min_coords[0]:.4g}, {max_coords[0]:.4g}], "
        f"z=[{min_coords[1]:.4g}, {max_coords[1]:.4g}]"
    )
    print(
        f"  Extended boundary: x=[{x_min:.4g}, {x_max:.4g}], "
        f"z=[{z_min:.4g}, {z_max:.4g}] (padding={padding:.4g})"
    )

    # Boundary square vertices (appended after original points)
    boundary_points = np.array(
        [[x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]]
    )

    # Combine: original points first, then boundary corners
    all_points = np.vstack([points_2d, boundary_points])

    # Boundary segments (connecting the 4 corners)
    b = n_original  # index offset for boundary vertices
    boundary_segments = np.array(
        [[b, b + 1], [b + 1, b + 2], [b + 2, b + 3], [b + 3, b]]
    )

    # Prepare input for the triangle library
    tri_input = {
        "vertices": all_points,
        "segments": boundary_segments,
    }

    # Triangulation flags:
    #   p  = triangulate a PSLG (Planar Straight Line Graph)
    #   q30 = quality mesh with minimum angle 30 degrees
    #   a  = maximum area constraint
    #   Q  = quiet mode
    flags = f"pq30a{max_area:.10g}Q"
    print(f"  Triangle flags: {flags}")

    tri_output = tr.triangulate(tri_input, flags)

    mesh_vertices = tri_output["vertices"]
    mesh_triangles = tri_output["triangles"]

    # Verify that the original vertices are preserved at the same indices
    original_check = np.allclose(mesh_vertices[:n_original], points_2d, atol=1e-12)
    if not original_check:
        print(
            "  Warning: Original vertex positions shifted slightly during triangulation."
        )
        # Force the original positions back
        mesh_vertices[:n_original] = points_2d

    print(
        f"  Triangulation: {len(mesh_vertices)} vertices, {len(mesh_triangles)} triangles"
    )

    return mesh_vertices, mesh_triangles


def _expand_boundary_locally(points, segments, n_original):
    """Add exterior CDT triangles along minimum-area paths to unused points.

    The convex hull is used only for finding candidate patches. The final
    domain retains the original interior and just the selected exterior patches.
    Boundary edges inside an added patch cease to be domain constraints; retaining
    them would force very thin triangles between the old and expanded boundaries.
    """
    coarse = tr.triangulate(
        {"vertices": points, "segments": segments,
         "segment_markers": np.full((len(segments), 1), 2, dtype=np.int32)},
        "pcQ",
    )
    vertices = coarse["vertices"]
    faces = coarse.get("triangles")
    if faces is None or not len(faces):
        raise ValueError("Boundary does not define a triangulatable domain")
    if not np.array_equal(vertices[:n_original], points[:n_original]):
        raise RuntimeError("Triangle did not preserve the original vertex prefix")

    # Marker 2 distinguishes original constraints from convex-hull segments
    # introduced by 'c', even when Triangle splits a constraint at an input point.
    constraints = {
        tuple(sorted(edge))
        for edge, marker in zip(coarse["segments"], coarse["segment_markers"].ravel())
        if marker == 2
    }
    edge_faces = {}
    incident = [[] for _ in vertices]
    for i, face in enumerate(faces):
        for v in face:
            incident[v].append(i)
        for a, b in zip(face, np.roll(face, -1)):
            edge_faces.setdefault(tuple(sorted((a, b))), []).append(i)

    neighbors = [[] for _ in faces]
    exterior = np.zeros(len(faces), dtype=bool)
    stack = []
    for edge, adjacent in edge_faces.items():
        if len(adjacent) == 2:
            a, b = adjacent
            neighbors[a].append((b, edge in constraints))
            neighbors[b].append((a, edge in constraints))
        elif edge not in constraints:
            stack.append(adjacent[0])
    # Flood from the hull, stopping at the supplied boundary.
    while stack:
        i = stack.pop()
        if exterior[i]:
            continue
        exterior[i] = True
        stack.extend(j for j, blocked in neighbors[i] if not blocked and not exterior[j])
    selected = ~exterior
    if not selected.any():
        raise ValueError("Boundary must contain a closed, nondegenerate domain")

    used = np.zeros(len(vertices), dtype=bool)
    used[faces[selected].ravel()] = True
    missing = np.flatnonzero(~used[:n_original])
    if not len(missing):
        return vertices, np.asarray(sorted(constraints), dtype=np.int32)

    # Multi-source Dijkstra: the cost is added area, rather than distance,
    # so narrow local patches are preferred over filling entire concavities.
    abc = vertices[faces]
    u, v = abc[:, 1] - abc[:, 0], abc[:, 2] - abc[:, 0]
    areas = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    distance = np.full(len(faces), np.inf)
    parent = np.full(len(faces), -1, dtype=int)
    roots = np.flatnonzero(selected)
    distance[roots] = 0
    queue = [(0.0, int(i)) for i in roots]
    heapq.heapify(queue)
    while queue:
        cost, i = heapq.heappop(queue)
        if cost > distance[i]:
            continue
        for j, _ in neighbors[i]:
            candidate = cost + areas[j]
            if candidate < distance[j]:
                distance[j], parent[j] = candidate, i
                heapq.heappush(queue, (candidate, j))
    for index in missing:
        if not incident[index]:
            raise ValueError(f"Original vertex {index} cannot participate in a triangle")
        i = min(incident[index], key=lambda j: distance[j])
        if not np.isfinite(distance[i]):
            raise ValueError(f"Cannot connect original vertex {index} to the domain")
        while not selected[i]:
            selected[i] = True
            i = parent[i]

    rim = {
        edge for edge, adjacent in edge_faces.items()
        if sum(bool(selected[i]) for i in adjacent) == 1
    }
    print(f"  Expanded boundary locally to include {len(missing)} exterior input vertices")
    return vertices, np.asarray(sorted(rim), dtype=np.int32)


def build_2d_triangulation_with_boundary(
    points_2d, n_original, max_area, boundary_verts_2d, boundary_segments
):
    """
    Build a Delaunay triangulation of the 2D points using an explicit
    boundary from an external .obj file. Expand the domain locally when
    necessary so every original vertex is referenced by a triangle.
    Original coordinates must be distinct; their order and values are preserved.

    Args:
        points_2d: Nx2 array of (x, z) coordinates from the input .obj.
        n_original: Number of original vertices to preserve.
        max_area: Maximum triangle area constraint for mesh refinement.
        boundary_verts_2d: Bx2 array of boundary vertex positions.
        boundary_segments: list of (i, j) 0-indexed boundary edge tuples
                           (indices into boundary_verts_2d).

    Returns:
        mesh_vertices_2d: Mx2 array of all mesh vertices (original first).
        mesh_triangles: Kx3 array of triangle vertex indices.
    """
    points_2d = np.asarray(points_2d, dtype=float)
    boundary_verts_2d = np.asarray(boundary_verts_2d, dtype=float)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2 or not len(points_2d):
        raise ValueError("points_2d must be a nonempty Nx2 array")
    if n_original != len(points_2d):
        raise ValueError("n_original must equal len(points_2d)")
    if boundary_verts_2d.ndim != 2 or boundary_verts_2d.shape[1] != 2:
        raise ValueError("boundary_verts_2d must be a Bx2 array")
    if not np.isfinite(points_2d).all() or not np.isfinite(boundary_verts_2d).all():
        raise ValueError("Vertex coordinates must be finite")
    if not np.isfinite(max_area) or max_area <= 0:
        raise ValueError("max_area must be positive and finite")
    if len(np.unique(points_2d, axis=0)) != n_original:
        raise ValueError("Duplicate original coordinates cannot all participate in a nondegenerate triangulation")
    boundary_segments = np.asarray(boundary_segments)
    if (boundary_segments.ndim != 2 or boundary_segments.shape[1] != 2
            or not len(boundary_segments)
            or not np.issubdtype(boundary_segments.dtype, np.integer)
            or np.any(boundary_segments < 0)
            or np.any(boundary_segments >= len(boundary_verts_2d))):
        raise ValueError("boundary_segments must contain valid boundary vertex index pairs")

    n_boundary = len(boundary_verts_2d)
    print(
        f"  Using external boundary: {n_boundary} vertices, "
        f"{len(boundary_segments)} segments"
    )

    # Reuse original indices for coincident boundary vertices. Otherwise Triangle
    # may reference the boundary copy and leave the original index unused.
    all_points = points_2d.tolist()
    indices = {tuple(p): i for i, p in enumerate(points_2d)}
    boundary_indices = []
    for p in boundary_verts_2d:
        key = tuple(p)
        if key not in indices:
            indices[key] = len(all_points)
            all_points.append(p.tolist())
        boundary_indices.append(indices[key])
    offset_segments = np.asarray(boundary_indices)[boundary_segments]
    if np.any(offset_segments[:, 0] == offset_segments[:, 1]):
        raise ValueError("Boundary contains a zero-length segment")
    all_points, offset_segments = _expand_boundary_locally(
        np.asarray(all_points), offset_segments, n_original
    )

    # Prepare input for the triangle library
    tri_input = {
        "vertices": all_points,
        "segments": offset_segments,
    }

    # Triangulation flags:
    #   p  = triangulate a PSLG (Planar Straight Line Graph)
    #   q30 = quality mesh with minimum angle 30 degrees
    #   a  = maximum area constraint
    #   Q  = quiet mode
    flags = f"pq30a{max_area:.10g}Q"
    print(f"  Triangle flags: {flags}")

    tri_output = tr.triangulate(tri_input, flags)

    mesh_vertices = tri_output["vertices"]
    mesh_triangles = tri_output["triangles"]

    if not np.array_equal(mesh_vertices[:n_original], points_2d):
        raise RuntimeError("Triangle did not preserve the original vertex prefix")
    used = np.zeros(len(mesh_vertices), dtype=bool)
    used[mesh_triangles.ravel()] = True
    missing = np.flatnonzero(~used[:n_original])
    if len(missing):
        raise RuntimeError(f"Original vertices are unused by triangles: {missing.tolist()}")

    print(
        f"  Triangulation: {len(mesh_vertices)} vertices, {len(mesh_triangles)} triangles"
    )

    return mesh_vertices, mesh_triangles

def main():
    parser = argparse.ArgumentParser(
        description="Generate a thin-shell mesh from 2D .obj points via Delaunay triangulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input .obj file with 2D vertices (y=0)")
    parser.add_argument("output", help="Output .obj file (extruded thin shell)")
    parser.add_argument(
        "--boundary",
        type=str,
        default=None,
        help="Optional boundary .obj file with vertices and line edges "
        "defining the domain boundary. If provided, the bounding "
        "box is NOT generated; the boundary from this file is used "
        "instead.",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=0.01,
        help="Maximum triangle area for mesh refinement (default: 0.01). "
        "Smaller values produce finer meshes.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Boundary padding as a fraction of bounding-box size (default: 0.1). "
        "Ignored when --boundary is specified.",
    )

    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    vertices_3d, obj_lines = read_obj_vertices(args.input)
    n_original = len(vertices_3d)
    print(f"  {n_original} input vertices, {len(obj_lines)} line elements")

    # Project to 2D (XZ plane, drop y)
    points_2d = vertices_3d[:, [0, 2]]

    print("Triangulating ...")
    if args.boundary:
        print(f"Reading boundary from {args.boundary} ...")
        boundary_verts_2d, boundary_segments = read_boundary_obj(args.boundary)
        mesh_verts_2d, mesh_tris = build_2d_triangulation_with_boundary(
            points_2d, n_original, args.max_area, boundary_verts_2d, boundary_segments
        )
    else:
        mesh_verts_2d, mesh_tris = build_2d_triangulation(
            points_2d, n_original, args.max_area, args.padding
        )

    # Write the 2D triangulation as a flat .obj (vertices: x, 0, -z)
    base, ext = (
        (args.output[: -len(".obj")], ".obj")
        if args.output.lower().endswith(".obj")
        else (args.output, "")
    )
    flat_obj_path = base + "_2d" + ext
    print(f"Writing flat 2D triangulation to {flat_obj_path} ...")
    flat_verts_3d = np.zeros((len(mesh_verts_2d), 3))
    flat_verts_3d[:, 0] = mesh_verts_2d[:, 0]  # x
    flat_verts_3d[:, 1] = 0.0  # y = 0
    flat_verts_3d[:, 2] = mesh_verts_2d[:, 1]  # z = y_2d
    # Swap column 0 and 1 of mesh_tris to match OBJ face orientation (counter-clockwise)
    mesh_tris_swapped = mesh_tris[:, [1, 0, 2]]
    write_obj(args.output, flat_verts_3d, mesh_tris_swapped)
    # write_obj(args.output, flat_verts_3d, mesh_tris)

    # Summary
    print(
        f"\nDone! Original vertex indices [1..{n_original}] are preserved on the top layer,"
    )
    print(
        f"      and [{n_original + 1 + (len(mesh_verts_2d) - n_original)}..{n_original + len(mesh_verts_2d)}]"
        f" on the bottom layer (shifted by {len(mesh_verts_2d)})."
    )


if __name__ == "__main__":
    main()
