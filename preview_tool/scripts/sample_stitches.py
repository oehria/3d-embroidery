#!/usr/bin/env python3
"""
sample_stitches.py

Reads an SVG file containing groups of lines and paths, filters for
blue or black strokes, merges consecutive lines into polylines,
samples each polyline at fixed intervals, and exports as .obj.

Supports stroke specified on groups or on individual child elements,
and color formats: hex (#RRGGBB), rgb(r,g,b), and named colors.

Usage:
    python sample_stitches.py input.svg -o output.obj --interval 5.0
"""

import argparse
import xml.etree.ElementTree as ET
import re
import math
import numpy as np
from collections import defaultdict


# Named colors we care about
NAMED_COLORS = {
    "black": (0, 0, 0),
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "navy": (0, 0, 128),
    "darkblue": (0, 0, 139),
    "royalblue": (65, 105, 225),
    "dodgerblue": (30, 144, 255),
}


def parse_stroke_color(stroke_str):
    """
    Parse a stroke color string and return (r, g, b) tuple.
    Supports:
      - Hex: #RRGGBB, #RGB
      - RGB function: rgb(r, g, b)
      - Named colors: 'black', 'blue', etc.
    Returns None if the string cannot be parsed.
    """
    if not stroke_str:
        return None
    stroke_str = stroke_str.strip()

    # Handle hex colors: #RRGGBB
    m = re.match(r"^#([0-9a-fA-F]{6})$", stroke_str)
    if m:
        hex_str = m.group(1)
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b)

    # Handle 3-digit hex: #RGB
    m = re.match(r"^#([0-9a-fA-F]{3})$", stroke_str)
    if m:
        hex_str = m.group(1)
        r = int(hex_str[0], 16) * 17
        g = int(hex_str[1], 16) * 17
        b = int(hex_str[2], 16) * 17
        return (r, g, b)

    # Handle rgb(r, g, b)
    m = re.match(r"^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", stroke_str)
    if m:
        r = int(m.group(1))
        g = int(m.group(2))
        b = int(m.group(3))
        return (r, g, b)

    # Handle named colors
    name = stroke_str.lower()
    if name in NAMED_COLORS:
        return NAMED_COLORS[name]

    return None


def is_blue_or_black(rgb):
    """
    Check if an (r, g, b) color is blue (blue component > 200) or black
    (all components <= 50).
    """
    if rgb is None:
        return False
    r, g, b = rgb
    # Blue: blue component > 200
    if b > 200:
        return True
    # Black: all components are very low
    if r <= 50 and g <= 50 and b <= 50:
        return True
    return False


def get_stroke_from_element(elem):
    """
    Extract stroke color string from an SVG element.
    Checks both the 'stroke' attribute and the 'style' attribute.
    Supports 'stroke: #hex', 'stroke: rgb(r,g,b)', and 'stroke: name'.
    Returns the parsed (r, g, b) tuple or None.
    """
    stroke = elem.get("stroke", "")
    if stroke:
        color = parse_stroke_color(stroke)
        if color is not None:
            return color

    style = elem.get("style", "")
    if style:
        # Match stroke in style: 'stroke: <value>'
        # Value can be hex, rgb(...), or a named color
        m = re.search(r"stroke\s*:\s*([^;]+)", style)
        if m:
            color = parse_stroke_color(m.group(1).strip())
            if color is not None:
                return color

    return None


def parse_path_to_segments(d_str, transform_matrix, uu_to_mm):
    """
    Parse an SVG path 'd' attribute and extract line segments.
    Supports M/m (moveto), L/l (lineto), H/h, V/v, and Z/z commands.
    Returns a list of (x1_mm, y1_mm, x2_mm, y2_mm) segments.
    """
    if not d_str:
        return []

    segments = []
    # Tokenize: split into commands and numbers
    tokens = re.findall(
        r"[MmLlHhVvZzCcSsQqTtAa]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d_str
    )

    current_x, current_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0  # for Z command
    cmd = None
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token in "MmLlHhVvZzCcSsQqTtAa":
            cmd = token
            i += 1
            continue

        if cmd is None:
            i += 1
            continue

        if cmd == "M":
            current_x = float(token)
            current_y = float(tokens[i + 1]) if i + 1 < len(tokens) else 0.0
            start_x, start_y = current_x, current_y
            i += 2
            cmd = "L"  # subsequent coordinates are implicit LineTo
        elif cmd == "m":
            current_x += float(token)
            current_y += float(tokens[i + 1]) if i + 1 < len(tokens) else 0.0
            start_x, start_y = current_x, current_y
            i += 2
            cmd = "l"
        elif cmd == "L":
            new_x = float(token)
            new_y = float(tokens[i + 1]) if i + 1 < len(tokens) else 0.0
            tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
            tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
            segments.append(
                (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
            )
            current_x, current_y = new_x, new_y
            i += 2
        elif cmd == "l":
            dx = float(token)
            dy = float(tokens[i + 1]) if i + 1 < len(tokens) else 0.0
            new_x = current_x + dx
            new_y = current_y + dy
            tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
            tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
            segments.append(
                (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
            )
            current_x, current_y = new_x, new_y
            i += 2
        elif cmd == "H":
            new_x = float(token)
            tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
            tx2, ty2 = apply_transform(transform_matrix, new_x, current_y)
            segments.append(
                (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
            )
            current_x = new_x
            i += 1
        elif cmd == "h":
            dx = float(token)
            new_x = current_x + dx
            tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
            tx2, ty2 = apply_transform(transform_matrix, new_x, current_y)
            segments.append(
                (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
            )
            current_x = new_x
            i += 1
        elif cmd == "V":
            new_y = float(token)
            tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
            tx2, ty2 = apply_transform(transform_matrix, current_x, new_y)
            segments.append(
                (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
            )
            current_y = new_y
            i += 1
        elif cmd == "v":
            dy = float(token)
            new_y = current_y + dy
            tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
            tx2, ty2 = apply_transform(transform_matrix, current_x, new_y)
            segments.append(
                (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
            )
            current_y = new_y
            i += 1
        elif cmd in ("Z", "z"):
            if abs(current_x - start_x) > 1e-9 or abs(current_y - start_y) > 1e-9:
                tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                tx2, ty2 = apply_transform(transform_matrix, start_x, start_y)
                segments.append(
                    (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                )
            current_x, current_y = start_x, start_y
            i += 1
        elif cmd in ("C", "c", "S", "s", "Q", "q", "T", "t", "A", "a"):
            # For curves: approximate by connecting the current point to the endpoint
            # Skip control points
            if cmd == "C":  # 6 params: x1,y1 x2,y2 x,y
                if i + 5 < len(tokens):
                    new_x, new_y = float(tokens[i + 4]), float(tokens[i + 5])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 6
            elif cmd == "c":  # relative cubic
                if i + 5 < len(tokens):
                    new_x = current_x + float(tokens[i + 4])
                    new_y = current_y + float(tokens[i + 5])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 6
            elif cmd in ("S", "Q"):  # 4 params
                if i + 3 < len(tokens):
                    new_x, new_y = float(tokens[i + 2]), float(tokens[i + 3])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 4
            elif cmd in ("s", "q"):  # relative, 4 params
                if i + 3 < len(tokens):
                    new_x = current_x + float(tokens[i + 2])
                    new_y = current_y + float(tokens[i + 3])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 4
            elif cmd == "T":  # 2 params
                if i + 1 < len(tokens):
                    new_x, new_y = float(token), float(tokens[i + 1])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 2
            elif cmd == "t":  # relative, 2 params
                if i + 1 < len(tokens):
                    new_x = current_x + float(token)
                    new_y = current_y + float(tokens[i + 1])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 2
            elif cmd == "A":  # 7 params
                if i + 6 < len(tokens):
                    new_x, new_y = float(tokens[i + 5]), float(tokens[i + 6])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 7
            elif cmd == "a":  # relative, 7 params
                if i + 6 < len(tokens):
                    new_x = current_x + float(tokens[i + 5])
                    new_y = current_y + float(tokens[i + 6])
                    tx1, ty1 = apply_transform(transform_matrix, current_x, current_y)
                    tx2, ty2 = apply_transform(transform_matrix, new_x, new_y)
                    segments.append(
                        (tx1 * uu_to_mm, ty1 * uu_to_mm, tx2 * uu_to_mm, ty2 * uu_to_mm)
                    )
                    current_x, current_y = new_x, new_y
                i += 7
            else:
                i += 1
        else:
            i += 1

    return segments


def parse_transform(transform_str):
    """
    Parse an SVG transform attribute and return a 3x3 affine matrix.
    Supports: matrix(a,b,c,d,e,f), translate(tx[,ty]), scale(sx[,sy]).
    """
    if not transform_str:
        return np.eye(3)

    result = np.eye(3)

    # Find all transform functions
    for m in re.finditer(r"(\w+)\s*\(([^)]+)\)", transform_str):
        func = m.group(1)
        args = [float(x) for x in re.split(r"[\s,]+", m.group(2).strip())]

        if func == "matrix" and len(args) == 6:
            a, b, c, d, e, f = args
            mat = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
            result = result @ mat
        elif func == "translate":
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0
            mat = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            result = result @ mat
        elif func == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            mat = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            result = result @ mat

    return result


def apply_transform(matrix, x, y):
    """Apply a 3x3 affine transform to a 2D point."""
    pt = np.array([x, y, 1.0])
    result = matrix @ pt
    return result[0], result[1]


def compute_svg_to_mm(root):
    """
    Compute the scale factor from SVG user units to millimeters,
    based on the width/height attributes and the viewBox.
    """
    ns = {"svg": "http://www.w3.org/2000/svg"}

    width_str = root.get("width", "")
    height_str = root.get("height", "")
    viewbox_str = root.get("viewBox", "")

    if not viewbox_str:
        # Default: assume 96dpi => 1 user unit = 1/96 inch = 0.26458 mm
        return 0.26458

    vb_parts = viewbox_str.split()
    vb_width = float(vb_parts[2])
    vb_height = float(vb_parts[3])

    def parse_length_mm(s):
        """Parse a length string with units and return value in mm."""
        s = s.strip()
        if s.endswith("cm"):
            return float(s[:-2]) * 10.0
        elif s.endswith("mm"):
            return float(s[:-2])
        elif s.endswith("in"):
            return float(s[:-2]) * 25.4
        elif s.endswith("pt"):
            return float(s[:-2]) * 25.4 / 72.0
        elif s.endswith("px"):
            return float(s[:-2]) * 25.4 / 96.0
        else:
            # No unit: treat as px (96 dpi)
            try:
                return float(s) * 25.4 / 96.0
            except ValueError:
                return None

    width_mm = parse_length_mm(width_str) if width_str else None
    height_mm = parse_length_mm(height_str) if height_str else None

    if width_mm and vb_width > 0:
        return width_mm / vb_width
    elif height_mm and vb_height > 0:
        return height_mm / vb_height
    else:
        return 0.26458  # fallback


def merge_lines_into_polylines(segments, tol=1e-9):
    """
    Given a list of line segments [(x1,y1,x2,y2), ...],
    merge segments that share endpoints into polylines.

    Returns a list of polylines, where each polyline is a list of (x, y) points.
    """
    if not segments:
        return []

    # Build an adjacency structure using endpoint coordinates as keys
    # Each segment becomes two endpoints; we want to chain them.

    # For efficient lookup, round coordinates to avoid floating-point issues
    def key(x, y):
        return (round(x, 4), round(y, 4))

    # Build adjacency: for each endpoint, store which segments connect there
    adj = defaultdict(list)
    for i, (x1, y1, x2, y2) in enumerate(segments):
        # print("i, x1, y1, x2, y2", i, x1, y1, x2, y2)
        k1 = key(x1, y1)
        k2 = key(x2, y2)
        adj[k1].append((i, "start"))
        adj[k2].append((i, "end"))

    used = set()
    polylines = []

    for seg_idx in range(len(segments)):
        if seg_idx in used:
            continue

        used.add(seg_idx)
        x1, y1, x2, y2 = segments[seg_idx]
        # Build polyline in both directions from this segment
        # Forward: from end of current segment
        chain = [(x1, y1), (x2, y2)]

        # Extend forward (from the end point)
        while True:
            last = key(chain[-1][0], chain[-1][1])
            found = False
            for idx, which_end in adj[last]:
                if idx in used:
                    continue
                sx1, sy1, sx2, sy2 = segments[idx]
                if which_end == "start" and key(sx1, sy1) == last:
                    chain.append((sx2, sy2))
                    used.add(idx)
                    found = True
                    break
                elif which_end == "end" and key(sx2, sy2) == last:
                    chain.append((sx1, sy1))
                    used.add(idx)
                    found = True
                    break
            if not found:
                break

        # Extend backward (from the start point)
        while True:
            first = key(chain[0][0], chain[0][1])
            found = False
            for idx, which_end in adj[first]:
                if idx in used:
                    continue
                sx1, sy1, sx2, sy2 = segments[idx]
                if which_end == "end" and key(sx2, sy2) == first:
                    chain.insert(0, (sx1, sy1))
                    used.add(idx)
                    found = True
                    break
                elif which_end == "start" and key(sx1, sy1) == first:
                    chain.insert(0, (sx2, sy2))
                    used.add(idx)
                    found = True
                    break
            if not found:
                break

        polylines.append(chain)

    return polylines


def sample_polyline(polyline, interval):
    """
    Sample a polyline at fixed-length intervals.

    Args:
        polyline: list of (x, y) points
        interval: sampling interval in same units as coordinates

    Returns:
        list of (x, y) sampled points (always includes first and last point)
    """
    if len(polyline) < 2:
        return list(polyline)

    # Compute cumulative arc lengths
    cum_lengths = [0.0]
    for i in range(1, len(polyline)):
        dx = polyline[i][0] - polyline[i - 1][0]
        dy = polyline[i][1] - polyline[i - 1][1]
        seg_len = math.sqrt(dx * dx + dy * dy)
        cum_lengths.append(cum_lengths[-1] + seg_len)

    total_length = cum_lengths[-1]
    if total_length < 1e-12:
        return [polyline[0]]

    # Generate sample distances
    n_samples = max(1, int(math.ceil(total_length / interval)))
    sample_dists = [i * interval for i in range(n_samples)]
    # Always include the last point
    if sample_dists[-1] < total_length - 1e-12:
        sample_dists.append(total_length)
    else:
        sample_dists[-1] = total_length  # snap to end

    # Sample points by linear interpolation along the polyline
    sampled = []
    seg_idx = 0
    for d in sample_dists:
        # Find the segment containing distance d
        while seg_idx < len(cum_lengths) - 2 and cum_lengths[seg_idx + 1] < d - 1e-12:
            seg_idx += 1

        seg_start = cum_lengths[seg_idx]
        seg_end = cum_lengths[seg_idx + 1]
        seg_len = seg_end - seg_start

        if seg_len < 1e-12:
            sampled.append(polyline[seg_idx])
        else:
            t = (d - seg_start) / seg_len
            t = max(0.0, min(1.0, t))
            x = polyline[seg_idx][0] + t * (
                polyline[seg_idx + 1][0] - polyline[seg_idx][0]
            )
            y = polyline[seg_idx][1] + t * (
                polyline[seg_idx + 1][1] - polyline[seg_idx][1]
            )
            sampled.append((x, y))

    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Sample stitches from SVG and export as OBJ."
    )
    parser.add_argument("input", help="Input SVG file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output OBJ file (default: input with .obj extension)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Sampling interval in mm (default: 5.0)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.rsplit(".", 1)[0] + ".obj"

    # Parse SVG
    tree = ET.parse(args.input)
    root = tree.getroot()

    # Handle SVG namespace
    ns = {"svg": "http://www.w3.org/2000/svg"}
    svg_ns = "http://www.w3.org/2000/svg"

    # Compute user-unit to mm scale
    uu_to_mm = compute_svg_to_mm(root)
    uu_to_mm = 1.0
    print(f"SVG user-unit to mm scale: {uu_to_mm:.6f}")

    # Collect all blue/black line segments (transformed to SVG user units)
    all_segments = []
    # Collect all non-blue/non-black segments (boundary)
    boundary_segments = []

    # Find all groups (direct children and nested)
    for group in root.iter(f"{{{svg_ns}}}g"):
        # Get stroke color from the group itself
        group_color = get_stroke_from_element(group)

        # Get transform for this group
        transform_str = group.get("transform", "")
        transform_matrix = parse_transform(transform_str)
        print("Transform matrix", transform_matrix)

        # Process <line> elements in this group
        for line in group.iter(f"{{{svg_ns}}}line"):
            # Check child-level stroke first, fall back to group stroke
            elem_color = get_stroke_from_element(line)
            effective_color = elem_color if elem_color is not None else group_color

            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))

            # Apply group transform to get SVG user-unit coordinates
            tx1, ty1 = apply_transform(transform_matrix, x1, y1)
            tx2, ty2 = apply_transform(transform_matrix, x2, y2)

            # Convert to mm
            mx1 = tx1 * uu_to_mm
            my1 = ty1 * uu_to_mm
            mx2 = tx2 * uu_to_mm
            my2 = ty2 * uu_to_mm

            if is_blue_or_black(effective_color):
                all_segments.append((mx1, my1, mx2, my2))
            else:
                boundary_segments.append((mx1, my1, mx2, my2))

        # Process <path> elements in this group
        for path in group.iter(f"{{{svg_ns}}}path"):
            # Check child-level stroke first, fall back to group stroke
            elem_color = get_stroke_from_element(path)
            effective_color = elem_color if elem_color is not None else group_color

            d_str = path.get("d", "")
            path_segments = parse_path_to_segments(d_str, transform_matrix, uu_to_mm)

            if is_blue_or_black(effective_color):
                all_segments.extend(path_segments)
            else:
                boundary_segments.extend(path_segments)

        # Process <polyline> elements in this group
        for polyline_elem in group.iter(f"{{{svg_ns}}}polyline"):
            # Check child-level stroke first, fall back to group stroke
            elem_color = get_stroke_from_element(polyline_elem)
            effective_color = elem_color if elem_color is not None else group_color

            points_str = polyline_elem.get("points", "")
            if not points_str:
                continue

            # Parse points: "x1,y1 x2,y2 x3,y3 ..."
            # Points can be separated by spaces/commas, coordinates within a point by commas
            coords = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", points_str)
            pts = []
            for j in range(0, len(coords) - 1, 2):
                x = float(coords[j])
                y = float(coords[j + 1])
                tx, ty = apply_transform(transform_matrix, x, y)
                pts.append((tx * uu_to_mm, ty * uu_to_mm))

            # Create segments from consecutive points
            seg_list = []
            for j in range(len(pts) - 1):
                seg_list.append((pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1]))

            if is_blue_or_black(effective_color):
                all_segments.extend(seg_list)
            else:
                boundary_segments.extend(seg_list)

    print(f"Total blue/black segments: {len(all_segments)}")
    print(f"Total boundary (non-blue/black) segments: {len(boundary_segments)}")

    # Merge into polylines
    polylines = merge_lines_into_polylines(all_segments)
    print(f"Merged into {len(polylines)} polylines")

    # Sample each polyline
    interval = args.interval
    all_sampled = []
    for pl in polylines:
        sampled = sample_polyline(pl, interval)
        if len(sampled) >= 2:
            all_sampled.append(sampled)

    print(f"After sampling at {interval}mm: {len(all_sampled)} polylines")

    # Export as OBJ
    vertex_idx = 1  # OBJ is 1-indexed
    with open(args.output, "w") as f:
        f.write(f"# Sampled stitches from {args.input}\n")
        f.write(f"# Sampling interval: {interval} mm\n")
        f.write(f"# Number of polylines: {len(all_sampled)}\n\n")

        for polyline in all_sampled:
            # Write vertices
            start_idx = vertex_idx
            for x, y in polyline:
                f.write(f"v {x:.6f} 0.000000 {-y:.6f} \n")
                vertex_idx += 1

            # Write line segments connecting consecutive vertices
            for i in range(start_idx, vertex_idx - 1):
                f.write(f"l {i} {i + 1}\n")

            f.write("\n")

    total_vertices = vertex_idx - 1
    total_edges = sum(len(pl) - 1 for pl in all_sampled)
    print(
        f"Exported {total_vertices} vertices and {total_edges} edges to {args.output}"
    )

    # --- Export boundary curves as a separate OBJ ---
    if boundary_segments:
        boundary_polylines = merge_lines_into_polylines(boundary_segments)
        print(f"Boundary: merged into {len(boundary_polylines)} polylines")

        boundary_output = args.output.rsplit(".", 1)[0] + "_boundary.obj"
        bv_idx = 1  # OBJ is 1-indexed
        with open(boundary_output, "w") as f:
            f.write(f"# Boundary curves from {args.input}\n")
            f.write(f"# Number of polylines: {len(boundary_polylines)}\n\n")

            for polyline in boundary_polylines:
                start_idx = bv_idx
                for x, y in polyline:
                    f.write(f"v {x:.6f} 0.000000 {-y:.6f} \n")
                    bv_idx += 1

                # Write line segments connecting consecutive vertices
                for i in range(start_idx, bv_idx - 1):
                    f.write(f"l {i} {i + 1}\n")

                f.write("\n")

        bv_total = bv_idx - 1
        be_total = sum(len(pl) - 1 for pl in boundary_polylines)
        print(
            f"Exported {bv_total} boundary vertices and {be_total} edges to {boundary_output}"
        )
    else:
        print("No boundary (non-blue/black) curves found.")


if __name__ == "__main__":
    main()
