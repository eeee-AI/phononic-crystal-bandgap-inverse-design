#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Conn6 = Tuple[int, int, int, int, int, int]
Conn8 = Tuple[int, int, int, int, int, int, int, int]


@dataclass
class NodeSection:
    name: str
    nodes: Dict[int, Vec3]


@dataclass
class ElementSection:
    name: str
    element_type: str
    elements: Dict[int, Conn6]


@dataclass
class UnitCell:
    nodes: Dict[int, Vec3]
    elements: Dict[int, Conn6]
    node_set_name: str
    element_set_name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    y_grid: List[float]
    z_grid: List[float]

    @property
    def a(self) -> float:
        return self.xmax - self.xmin

    @property
    def thickness(self) -> float:
        return self.zmax - self.zmin


class NodeRegistry:
    def __init__(self, tolerance: float):
        self.tol = float(tolerance)
        if self.tol <= 0.0:
            raise ValueError("Node merge tolerance must be positive.")
        self.nodes: Dict[int, Vec3] = {}
        self._key_to_id: Dict[Tuple[int, int, int], int] = {}
        self._next_id = 1

    def _key(self, xyz: Vec3) -> Tuple[int, int, int]:
        return tuple(int(round(v / self.tol)) for v in xyz) 

    def add(self, xyz: Vec3) -> int:
        key = self._key(xyz)
        old = self._key_to_id.get(key)
        if old is not None:
            ox, oy, oz = self.nodes[old]
            if max(abs(ox - xyz[0]), abs(oy - xyz[1]), abs(oz - xyz[2])) > 2.0 * self.tol:
                raise ValueError("Coordinate hashing collision; reduce --merge-tolerance.")
            return old
        nid = self._next_id
        self._next_id += 1
        self._key_to_id[key] = nid
        self.nodes[nid] = xyz
        return nid


class ModelBuilder:
    def __init__(self, tolerance: float):
        self.registry = NodeRegistry(tolerance)
        self.elements: Dict[int, Conn6] = {}
        self.elsets: Dict[str, List[int]] = defaultdict(list)
        self._next_eid = 1

    @property
    def nodes(self) -> Dict[int, Vec3]:
        return self.registry.nodes

    def add_element(self, conn: Sequence[int], set_names: Iterable[str]) -> int:
        if len(conn) != 6:
            raise ValueError("C3D6 connectivity must contain six nodes.")
        eid = self._next_eid
        self._next_eid += 1
        c = tuple(int(v) for v in conn)
        self.elements[eid] = c
        for name in set_names:
            self.elsets[name].append(eid)
        return eid


def parse_keyword(line: str) -> Tuple[str, Dict[str, str], List[str]]:
    fields = [part.strip() for part in line.strip().split(",")]
    keyword = fields[0].lstrip("*").strip().upper()
    options: Dict[str, str] = {}
    flags: List[str] = []
    for field in fields[1:]:
        if not field:
            continue
        if "=" in field:
            key, value = field.split("=", 1)
            options[key.strip().upper()] = value.strip()
        else:
            flags.append(field.upper())
    return keyword, options, flags


def read_mesh_sections(path: Path) -> Tuple[List[NodeSection], List[ElementSection]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    node_sections: List[NodeSection] = []
    element_sections: List[ElementSection] = []

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw or raw.startswith("**"):
            i += 1
            continue
        if not raw.startswith("*"):
            i += 1
            continue

        keyword, options, _ = parse_keyword(raw)
        if keyword == "NODE":
            name = options.get("NSET", f"NODE_SECTION_{len(node_sections) + 1}")
            data: Dict[int, Vec3] = {}
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                row = lines[i].strip()
                if row and not row.startswith("**"):
                    cols = [v.strip() for v in row.split(",")]
                    if len(cols) >= 4:
                        data[int(cols[0])] = (float(cols[1]), float(cols[2]), float(cols[3]))
                i += 1
            node_sections.append(NodeSection(name=name, nodes=data))
            continue

        if keyword == "ELEMENT":
            etype = options.get("TYPE", "").upper()
            name = options.get("ELSET", f"ELEMENT_SECTION_{len(element_sections) + 1}")
            data_e: Dict[int, Conn6] = {}
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                row = lines[i].strip()
                if row and not row.startswith("**"):
                    cols = [v.strip() for v in row.split(",") if v.strip()]
                    if len(cols) >= 7:
                        eid = int(cols[0])
                        conn = tuple(int(v) for v in cols[1:7])
                        data_e[eid] = conn  
                i += 1
            element_sections.append(ElementSection(name=name, element_type=etype, elements=data_e))
            continue

        i += 1

    return node_sections, element_sections


def choose_section_by_name(items, requested: Optional[str], attr: str):
    if requested is None:
        return None
    target = requested.lower()
    for item in items:
        if getattr(item, attr).lower() == target:
            return item
    available = ", ".join(getattr(item, attr) for item in items)
    raise ValueError(f"Requested section '{requested}' was not found. Available: {available}")


def load_unit_cell(
    path: Path,
    requested_node_set: Optional[str],
    requested_element_set: Optional[str],
    tolerance: float,
) -> UnitCell:
    node_sections, element_sections = read_mesh_sections(path)
    if not node_sections:
        raise ValueError("No *NODE section was found.")
    c3d6_sections = [s for s in element_sections if s.element_type.upper() == "C3D6"]
    if not c3d6_sections:
        raise ValueError("No *ELEMENT, TYPE=C3D6 section was found.")

    elem_sec = choose_section_by_name(c3d6_sections, requested_element_set, "name")
    if elem_sec is None:
        real_candidates = [s for s in c3d6_sections if "real" in s.name.lower()]
        elem_sec = real_candidates[0] if real_candidates else c3d6_sections[0]

    node_sec = choose_section_by_name(node_sections, requested_node_set, "name")
    required = {nid for conn in elem_sec.elements.values() for nid in conn}
    if node_sec is None:
        covering = [s for s in node_sections if required.issubset(s.nodes.keys())]
        if not covering:
            raise ValueError(
                f"No *NODE section contains all nodes referenced by element set '{elem_sec.name}'."
            )
        real_candidates = [s for s in covering if "real" in s.name.lower()]
        node_sec = real_candidates[0] if real_candidates else covering[0]
    elif not required.issubset(node_sec.nodes.keys()):
        missing = sorted(required.difference(node_sec.nodes.keys()))[:10]
        raise ValueError(
            f"Node set '{node_sec.name}' does not contain all element nodes; examples: {missing}"
        )

    used_nodes = {nid: node_sec.nodes[nid] for nid in sorted(required)}
    xs = [xyz[0] for xyz in used_nodes.values()]
    ys = [xyz[1] for xyz in used_nodes.values()]
    zs = [xyz[2] for xyz in used_nodes.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    a = xmax - xmin
    if a <= 0.0 or ymax <= ymin or zmax <= zmin:
        raise ValueError("The selected unit-cell mesh has invalid dimensions.")

    face_tol = max(tolerance, a * 1.0e-8)
    left_face = {
        nid: xyz for nid, xyz in used_nodes.items() if abs(xyz[0] - xmin) <= face_tol
    }
    right_face = {
        nid: xyz for nid, xyz in used_nodes.items() if abs(xyz[0] - xmax) <= face_tol
    }
    if not left_face or not right_face:
        raise ValueError("Could not identify the unit-cell left/right boundary nodes.")

    def uniq(values: Iterable[float]) -> List[float]:
        out: List[float] = []
        for value in sorted(values):
            if not out or abs(value - out[-1]) > face_tol:
                out.append(value)
        return out

    y_grid = uniq(xyz[1] for xyz in left_face.values())
    z_grid = uniq(xyz[2] for xyz in left_face.values())
    expected = len(y_grid) * len(z_grid)
    if len(left_face) != expected or len(right_face) != expected:
        raise ValueError(
            "The left/right unit-cell faces are not a full structured y-z tensor grid. "
            "A conforming structured transition/PML mesh cannot be generated automatically."
        )

    left_pairs = sorted((round(v[1] / face_tol), round(v[2] / face_tol)) for v in left_face.values())
    right_pairs = sorted((round(v[1] / face_tol), round(v[2] / face_tol)) for v in right_face.values())
    if left_pairs != right_pairs:
        raise ValueError("The left and right unit-cell faces do not have matching y-z meshes.")

    top = [v for v in used_nodes.values() if abs(v[1] - ymax) <= face_tol]
    bottom = [v for v in used_nodes.values() if abs(v[1] - ymin) <= face_tol]
    top_xz = sorted((round(v[0] / face_tol), round(v[2] / face_tol)) for v in top)
    bottom_xz = sorted((round(v[0] / face_tol), round(v[2] / face_tol)) for v in bottom)
    if top_xz != bottom_xz:
        raise ValueError("The unit-cell top and bottom faces do not have matching x-z meshes.")

    return UnitCell(
        nodes=used_nodes,
        elements=elem_sec.elements,
        node_set_name=node_sec.name,
        element_set_name=elem_sec.name,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        y_grid=y_grid,
        z_grid=z_grid,
    )


def linspace(start: float, end: float, intervals: int) -> List[float]:
    if intervals < 1:
        raise ValueError("The number of intervals must be at least one.")
    step = (end - start) / float(intervals)
    values = [start + i * step for i in range(intervals + 1)]
    values[0], values[-1] = start, end
    return values


def add_structured_wedge_region(
    model: ModelBuilder,
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    z_grid: Sequence[float],
    region_set: str,
    x_interval_sets: Optional[Sequence[str]] = None,
) -> List[List[int]]:
    nx = len(x_grid) - 1
    ny = len(y_grid) - 1
    nz = len(z_grid) - 1
    if min(nx, ny, nz) < 1:
        raise ValueError(f"Region '{region_set}' has an invalid grid.")
    if x_interval_sets is not None and len(x_interval_sets) != nx:
        raise ValueError("x_interval_sets length must equal the number of x intervals.")

    layer_eids: List[List[int]] = [[] for _ in range(nx)]

    node_ids: Dict[Tuple[int, int, int], int] = {}
    for ix, x in enumerate(x_grid):
        for iy, y in enumerate(y_grid):
            for iz, z in enumerate(z_grid):
                node_ids[(ix, iy, iz)] = model.registry.add((x, y, z))

    for ix in range(nx):
        interval_set = x_interval_sets[ix] if x_interval_sets is not None else None
        for iy in range(ny):
            for iz in range(nz):
                b00 = node_ids[(ix, iy, iz)]
                b10 = node_ids[(ix + 1, iy, iz)]
                b11 = node_ids[(ix + 1, iy + 1, iz)]
                b01 = node_ids[(ix, iy + 1, iz)]
                t00 = node_ids[(ix, iy, iz + 1)]
                t10 = node_ids[(ix + 1, iy, iz + 1)]
                t11 = node_ids[(ix + 1, iy + 1, iz + 1)]
                t01 = node_ids[(ix, iy + 1, iz + 1)]

                if (ix + iy) % 2 == 0:
                    wedges = [
                        (b00, b10, b11, t00, t10, t11),
                        (b00, b11, b01, t00, t11, t01),
                    ]
                else:
                    wedges = [
                        (b00, b10, b01, t00, t10, t01),
                        (b10, b11, b01, t10, t11, t01),
                    ]

                for conn in wedges:
                    sets = [region_set]
                    if interval_set:
                        sets.append(interval_set)
                    eid = model.add_element(conn, sets)
                    layer_eids[ix].append(eid)
    return layer_eids


def add_repeated_unit_cells(model: ModelBuilder, cell: UnitCell, repeats: int) -> None:
    if repeats < 1:
        raise ValueError("Unit-cell repeat count must be at least one.")

    for icell in range(repeats):
        local_to_global: Dict[int, int] = {}
        x_shift = icell * cell.a - cell.xmin
        z_shift = -cell.zmin
        for old_nid, (x, y, z) in cell.nodes.items():
            local_to_global[old_nid] = model.registry.add((x + x_shift, y, z + z_shift))

        cell_set = f"PNC_CELL_{icell + 1:03d}"
        for conn in cell.elements.values():
            new_conn = tuple(local_to_global[nid] for nid in conn)
            model.add_element(new_conn, ["PNC_ALL", cell_set])


def find_nodes_on_plane(
    nodes: Dict[int, Vec3],
    axis: int,
    value: float,
    tolerance: float,
    x_range: Optional[Tuple[float, float]] = None,
) -> List[int]:
    out = []
    for nid, xyz in nodes.items():
        if abs(xyz[axis] - value) > tolerance:
            continue
        if x_range is not None and not (
            x_range[0] - tolerance <= xyz[0] <= x_range[1] + tolerance
        ):
            continue
        out.append(nid)
    return sorted(out)


def periodic_pairs_y(
    nodes: Dict[int, Vec3],
    ymin: float,
    ymax: float,
    x_range: Tuple[float, float],
    tolerance: float,
) -> List[Tuple[int, int]]:
    bottom = find_nodes_on_plane(nodes, 1, ymin, tolerance, x_range=x_range)
    top = find_nodes_on_plane(nodes, 1, ymax, tolerance, x_range=x_range)

    def key(nid: int) -> Tuple[int, int]:
        x, _, z = nodes[nid]
        return int(round(x / tolerance)), int(round(z / tolerance))

    bmap = {key(nid): nid for nid in bottom}
    tmap = {key(nid): nid for nid in top}
    if set(bmap) != set(tmap):
        only_b = list(set(bmap).difference(tmap))[:5]
        only_t = list(set(tmap).difference(bmap))[:5]
        raise ValueError(
            f"Periodic top/bottom meshes do not match. bottom-only={only_b}, top-only={only_t}"
        )
    return [(tmap[k], bmap[k]) for k in sorted(bmap)]


def write_id_list(f, ids: Sequence[int], per_line: int = 16) -> None:
    for i in range(0, len(ids), per_line):
        f.write(", ".join(str(v) for v in ids[i : i + per_line]) + "\n")


def write_nset(f, name: str, ids: Sequence[int]) -> None:
    f.write(f"*NSET, NSET={name}\n")
    write_id_list(f, sorted(set(ids)))


def write_elset(f, name: str, ids: Sequence[int]) -> None:
    f.write(f"*ELSET, ELSET={name}\n")
    write_id_list(f, sorted(set(ids)))


def rayleigh_coefficients(zeta: float, fmin: float, fmax: float) -> Tuple[float, float]:
    if zeta <= 0.0:
        return 0.0, 0.0
    w1 = 2.0 * math.pi * fmin
    w2 = 2.0 * math.pi * fmax
    alpha = 2.0 * zeta * w1 * w2 / (w1 + w2)
    beta = 2.0 * zeta / (w1 + w2)
    return alpha, beta


def signed_wedge_measure(nodes: Dict[int, Vec3], conn: Conn6) -> float:
    p1, p2, p3, p4, _, _ = [nodes[nid] for nid in conn]
    ax, ay, az = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    bx, by, bz = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]
    cx, cy, cz = p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2]
    cross = (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
    return cross[0] * cx + cross[1] * cy + cross[2] * cz


def validate_model(model: ModelBuilder) -> None:
    node_ids = set(model.nodes)
    for eid, conn in model.elements.items():
        missing = set(conn).difference(node_ids)
        if missing:
            raise ValueError(f"Element {eid} references missing nodes: {sorted(missing)}")
        measure = signed_wedge_measure(model.nodes, conn)
        if measure <= 0.0:
            raise ValueError(f"Element {eid} has non-positive C3D6 orientation/volume measure.")



C3D6_FACE_NODE_INDEX = {
    "S1": (0, 1, 2),
    "S2": (3, 4, 5),
    "S3": (0, 1, 4, 3),
    "S4": (1, 2, 5, 4),
    "S5": (2, 0, 3, 5),
}


def c3d6_faces_on_x_plane(
    model: ModelBuilder,
    element_ids: Sequence[int],
    x_value: float,
    tolerance: float,
) -> List[Tuple[int, str]]:
    """Return the C3D6 element faces lying completely on x=x_value."""
    faces: List[Tuple[int, str]] = []
    for eid in element_ids:
        conn = model.elements[eid]
        matches: List[str] = []
        for face_name, local_ids in C3D6_FACE_NODE_INDEX.items():
            if all(
                abs(model.nodes[conn[i]][0] - x_value) <= tolerance
                for i in local_ids
            ):
                matches.append(face_name)
        if len(matches) > 1:
            raise ValueError(
                f"Expected at most one boundary face on element {eid} at "
                f"x={x_value:g}; found {matches}."
            )
        if matches:
            faces.append((eid, matches[0]))
    if not faces:
        raise ValueError(f"No C3D6 boundary faces were found at x={x_value:g}.")
    return faces


def build_cin3d8_layer(
    y_grid: Sequence[float],
    z_grid: Sequence[float],
    interface_x: float,
    side: str,
    first_node_id: int,
    first_element_id: int,
    geometry_mode: str,
    display_length: float,
    pole: Optional[Vec3] = None,
) -> dict:

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    if geometry_mode not in {"rectangular", "radial"}:
        raise ValueError("geometry_mode must be 'rectangular' or 'radial'.")
    if display_length <= 0.0:
        raise ValueError("CIN3D8 display length must be positive.")
    if geometry_mode == "radial" and pole is None:
        raise ValueError("Radial CIN3D8 geometry requires a pole.")
    if len(y_grid) < 2 or len(z_grid) < 2:
        raise ValueError("The CIN3D8 terminating surface needs a 2D y-z grid.")

    extra_nodes: Dict[int, Vec3] = {}
    near: Dict[Tuple[int, int], int] = {}
    far: Dict[Tuple[int, int], int] = {}
    next_nid = first_node_id

    outward_sign = -1.0 if side == "left" else 1.0

    for iy, y in enumerate(y_grid):
        for iz, z in enumerate(z_grid):
            near_id = next_nid
            next_nid += 1
            far_id = next_nid
            next_nid += 1

            near_xyz = (interface_x, y, z)

            if geometry_mode == "rectangular":
                far_xyz = (
                    interface_x + outward_sign * display_length,
                    y,
                    z,
                )
            else:
                assert pole is not None
                px, py, pz = pole
                far_xyz = (
                    2.0 * interface_x - px,
                    2.0 * y - py,
                    2.0 * z - pz,
                )

            if outward_sign * (far_xyz[0] - interface_x) <= 0.0:
                raise ValueError(
                    f"{side} CIN3D8 auxiliary nodes are not projected outward."
                )

            extra_nodes[near_id] = near_xyz
            extra_nodes[far_id] = far_xyz
            near[(iy, iz)] = near_id
            far[(iy, iz)] = far_id

    elements: Dict[int, Conn8] = {}
    next_eid = first_element_id
    ny = len(y_grid) - 1
    nz = len(z_grid) - 1

    for iy in range(ny):
        for iz in range(nz):
            if side == "left":
                corners = (
                    (iy, iz),
                    (iy, iz + 1),
                    (iy + 1, iz + 1),
                    (iy + 1, iz),
                )
            else:
                corners = (
                    (iy, iz),
                    (iy + 1, iz),
                    (iy + 1, iz + 1),
                    (iy, iz + 1),
                )

            conn = tuple(near[c] for c in corners) + tuple(far[c] for c in corners)
            elements[next_eid] = conn  
            next_eid += 1

    near_ids = sorted(near.values())
    far_ids = sorted(far.values())

    far_periodic_pairs = [
        (far[(len(y_grid) - 1, iz)], far[(0, iz)])
        for iz in range(len(z_grid))
    ]

    tol = 1.0e-9 * max(
        1.0,
        abs(interface_x),
        display_length,
        max(abs(v) for v in y_grid),
        max(abs(v) for v in z_grid),
    )

    if geometry_mode == "rectangular":
        expected_far_x = interface_x + outward_sign * display_length
        for key in near:
            nxyz = extra_nodes[near[key]]
            fxyz = extra_nodes[far[key]]
            if abs(fxyz[0] - expected_far_x) > tol:
                raise ValueError("Rectangular CIN3D8 far face is not planar.")
            if abs(fxyz[1] - nxyz[1]) > tol or abs(fxyz[2] - nxyz[2]) > tol:
                raise ValueError(
                    "Rectangular CIN3D8 must preserve the near-face y-z coordinates."
                )
    else:
        assert pole is not None
        for key in near:
            nxyz = extra_nodes[near[key]]
            fxyz = extra_nodes[far[key]]
            recovered_pole = tuple(
                2.0 * nxyz[i] - fxyz[i] for i in range(3)
            )
            if max(
                abs(recovered_pole[i] - pole[i]) for i in range(3)
            ) > tol:
                raise ValueError("Invalid CIN3D8 radial projection geometry.")

    return {
        "nodes": extra_nodes,
        "elements": elements,
        "near_ids": near_ids,
        "far_ids": far_ids,
        "far_periodic_pairs": far_periodic_pairs,
        "interface_x": interface_x,
        "geometry_mode": geometry_mode,
        "display_length": display_length,
        "pole": pole,
    }

def validate_cin3d8_layers(extra_nodes: Dict[int, Vec3], layers: Sequence[dict]) -> None:
    all_eids = set()
    for layer in layers:
        for eid, conn in layer["elements"].items():
            if eid in all_eids:
                raise ValueError(f"Duplicate CIN3D8 element label {eid}.")
            all_eids.add(eid)
            if len(conn) != 8:
                raise ValueError(f"CIN3D8 element {eid} does not have 8 nodes.")
            missing = set(conn).difference(extra_nodes)
            if missing:
                raise ValueError(
                    f"CIN3D8 element {eid} references missing nodes: {sorted(missing)}"
                )

def build_model(args) -> Tuple[ModelBuilder, UnitCell, dict]:
    unit_path = Path(args.unit_inp)
    cell = load_unit_cell(
        unit_path,
        requested_node_set=args.node_set,
        requested_element_set=args.element_set,
        tolerance=args.merge_tolerance,
    )
    a = cell.a
    if abs((cell.ymax - cell.ymin) - a) > max(
        args.merge_tolerance * 10.0, a * 1.0e-5
    ):
        raise ValueError(
            f"Fig. 8(a) expects a square cell, but dx={a:g} and "
            f"dy={cell.ymax-cell.ymin:g}."
        )

    tolerance = max(args.merge_tolerance, a * 1.0e-10)
    model = ModelBuilder(tolerance)

    transition_length = args.transition_length_cells * a
    infinite_display_length = args.infinite_display_length_cells * a
    pole_distance = args.infinite_pole_distance_cells * a
    pnc_length = args.repeats * a

    x_pnc_start = 0.0
    x_response = pnc_length
    x_left_interface = x_pnc_start - transition_length
    x_right_interface = x_response + transition_length
    x_excitation = (
        x_pnc_start if args.excitation_plane == 0 else x_left_interface
    )

    n_transition = max(1, int(math.ceil(transition_length / args.length)))
    z_grid = [z - cell.zmin for z in cell.z_grid]
    y_grid = list(cell.y_grid)
    y_mid = 0.5 * (cell.ymin + cell.ymax)
    z_mid = 0.5 * (z_grid[0] + z_grid[-1])

    left_transition_x = linspace(x_left_interface, x_pnc_start, n_transition)
    left_transition_interval_sets = [
        f"TRANSITION_LEFT_X_{i + 1:03d}" for i in range(n_transition)
    ]
    left_transition_layers = add_structured_wedge_region(
        model,
        left_transition_x,
        y_grid,
        z_grid,
        "TRANSITION",
        left_transition_interval_sets,
    )
    model.elsets["TRANSITION_LEFT"].extend(
        eid for layer in left_transition_layers for eid in layer
    )
    left_monitor_layer = (
        left_transition_layers[-1]
        if args.excitation_plane == 0
        else left_transition_layers[0]
    )
    model.elsets["MONITOR_LEFT"].extend(left_monitor_layer)

    add_repeated_unit_cells(model, cell, args.repeats)

    right_transition_x = linspace(x_response, x_right_interface, n_transition)
    right_transition_interval_sets = [
        f"TRANSITION_RIGHT_X_{i + 1:03d}" for i in range(n_transition)
    ]
    right_transition_layers = add_structured_wedge_region(
        model,
        right_transition_x,
        y_grid,
        z_grid,
        "TRANSITION",
        right_transition_interval_sets,
    )
    model.elsets["TRANSITION_RIGHT"].extend(
        eid for layer in right_transition_layers for eid in layer
    )
    model.elsets["MONITOR_RIGHT"].extend(right_transition_layers[0])

    validate_model(model)

    plane_tol = max(tolerance * 2.0, a * 1.0e-9)
    left_finite_faces = c3d6_faces_on_x_plane(
        model, left_transition_layers[0], x_left_interface, plane_tol
    )
    right_finite_faces = c3d6_faces_on_x_plane(
        model, right_transition_layers[-1], x_right_interface, plane_tol
    )

    left_pole = (
        (x_left_interface + pole_distance, y_mid, z_mid)
        if args.infinite_geometry == "radial"
        else None
    )
    right_pole = (
        (x_right_interface - pole_distance, y_mid, z_mid)
        if args.infinite_geometry == "radial"
        else None
    )

    first_extra_nid = max(model.nodes) + 1
    first_cin_eid = max(model.elements) + 1
    left_inf = build_cin3d8_layer(
        y_grid=y_grid,
        z_grid=z_grid,
        interface_x=x_left_interface,
        side="left",
        first_node_id=first_extra_nid,
        first_element_id=first_cin_eid,
        geometry_mode=args.infinite_geometry,
        display_length=infinite_display_length,
        pole=left_pole,
    )
    right_inf = build_cin3d8_layer(
        y_grid=y_grid,
        z_grid=z_grid,
        interface_x=x_right_interface,
        side="right",
        first_node_id=max(left_inf["nodes"]) + 1,
        first_element_id=max(left_inf["elements"]) + 1,
        geometry_mode=args.infinite_geometry,
        display_length=infinite_display_length,
        pole=right_pole,
    )

    extra_nodes: Dict[int, Vec3] = {}
    extra_nodes.update(left_inf["nodes"])
    extra_nodes.update(right_inf["nodes"])
    validate_cin3d8_layers(extra_nodes, (left_inf, right_inf))

    finite_pairs = periodic_pairs_y(
        model.nodes,
        cell.ymin,
        cell.ymax,
        (x_left_interface, x_right_interface),
        plane_tol,
    )
    pairs = list(finite_pairs)
    if not args.no_infinite_far_pbc:
        pairs.extend(left_inf["far_periodic_pairs"])
        pairs.extend(right_inf["far_periodic_pairs"])

    pbc_top_finite = [top for top, _ in finite_pairs]
    pbc_bottom_finite = [bottom for _, bottom in finite_pairs]
    pbc_top_set = set(pbc_top_finite)

    input_all = find_nodes_on_plane(model.nodes, 0, x_excitation, plane_tol)
    input_driven = [nid for nid in input_all if nid not in pbc_top_set]
    response_all = find_nodes_on_plane(model.nodes, 0, x_response, plane_tol)
    pnc_left_interface = find_nodes_on_plane(
        model.nodes, 0, x_pnc_start, plane_tol
    )
    left_inf_interface = find_nodes_on_plane(
        model.nodes, 0, x_left_interface, plane_tol
    )
    right_inf_interface = find_nodes_on_plane(
        model.nodes, 0, x_right_interface, plane_tol
    )

    transition_left_top = find_nodes_on_plane(
        model.nodes,
        1,
        cell.ymax,
        plane_tol,
        (x_left_interface, x_pnc_start),
    )
    transition_left_bottom = find_nodes_on_plane(
        model.nodes,
        1,
        cell.ymin,
        plane_tol,
        (x_left_interface, x_pnc_start),
    )
    transition_right_top = find_nodes_on_plane(
        model.nodes,
        1,
        cell.ymax,
        plane_tol,
        (x_response, x_right_interface),
    )
    transition_right_bottom = find_nodes_on_plane(
        model.nodes,
        1,
        cell.ymin,
        plane_tol,
        (x_response, x_right_interface),
    )
    transition_top = sorted(set(transition_left_top + transition_right_top))
    transition_bottom = sorted(
        set(transition_left_bottom + transition_right_bottom)
    )
    pnc_top = find_nodes_on_plane(
        model.nodes, 1, cell.ymax, plane_tol, (x_pnc_start, x_response)
    )
    pnc_bottom = find_nodes_on_plane(
        model.nodes, 1, cell.ymin, plane_tol, (x_pnc_start, x_response)
    )

    nsets = {
        "PBC_TOP": pbc_top_finite,
        "PBC_BOTTOM": pbc_bottom_finite,
        "TRANSITION_TOP": transition_top,
        "TRANSITION_BOTTOM": transition_bottom,
        "TRANSITION_LEFT_TOP": transition_left_top,
        "TRANSITION_LEFT_BOTTOM": transition_left_bottom,
        "TRANSITION_RIGHT_TOP": transition_right_top,
        "TRANSITION_RIGHT_BOTTOM": transition_right_bottom,
        "PNC_TOP": pnc_top,
        "PNC_BOTTOM": pnc_bottom,
        "INPUT_PLANE": input_all,
        "INPUT_DRIVEN": input_driven,
        "EXCITATION_PLANE": input_all,
        "PNC_LEFT_INTERFACE": pnc_left_interface,
        "RESPONSE_PLANE": response_all,
        "LEFT_INFINITY_INTERFACE_FINITE": left_inf_interface,
        "RIGHT_INFINITY_INTERFACE_FINITE": right_inf_interface,
        "LEFT_INFINITY_NEAR": left_inf["near_ids"],
        "LEFT_INFINITY_FAR": left_inf["far_ids"],
        "RIGHT_INFINITY_NEAR": right_inf["near_ids"],
        "RIGHT_INFINITY_FAR": right_inf["far_ids"],
    }

    surfaces = {
        "LEFT_TRANSITION_OUTER": left_finite_faces,
        "RIGHT_TRANSITION_OUTER": right_finite_faces,
        "LEFT_INFINITY_INTERFACE": [
            (eid, "S1") for eid in sorted(left_inf["elements"])
        ],
        "RIGHT_INFINITY_INTERFACE": [
            (eid, "S1") for eid in sorted(right_inf["elements"])
        ],
    }

    metadata = {
        "source_file": str(unit_path),
        "selected_node_set": cell.node_set_name,
        "selected_element_set": cell.element_set_name,
        "infinite_element_type": "CIN3D8",
        "repeats": args.repeats,
        "lattice_constant_mm": a,
        "plate_thickness_mm": cell.thickness,
        "transition_length_each_side_mm": transition_length,
        "requested_transition_x_length_mm": args.length,
        "actual_transition_x_length_mm": transition_length / n_transition,
        "structured_transition_intervals_each_side": n_transition,
        "infinite_geometry": args.infinite_geometry,
        "infinite_display_length_mm": infinite_display_length,
        "infinite_pole_distance_mm": (
            pole_distance if args.infinite_geometry == "radial" else None
        ),
        "left_infinite_pole_mm": left_pole,
        "right_infinite_pole_mm": right_pole,
        "left_infinite_far_face_x_mm": (
            x_left_interface - infinite_display_length
        ),
        "right_infinite_far_face_x_mm": (
            x_right_interface + infinite_display_length
        ),
        "input_displacement_mm": {
            "Ux": args.ux,
            "Uy": args.uy,
            "Uz": args.uz,
        },
        "excitation_plane_mode": args.excitation_plane,
        "excitation_plane_description": (
            "PnC entrance / left-transition interface"
            if args.excitation_plane == 0
            else "left-CIN3D8 / left-transition interface"
        ),
        "excitation_x_mm": x_excitation,
        "monitor_left_location": (
            "last left-transition layer adjacent to PnC entrance"
            if args.excitation_plane == 0
            else "first left-transition layer adjacent to CIN3D8 interface"
        ),
        "finite_node_count": len(model.nodes),
        "infinite_auxiliary_node_count": len(extra_nodes),
        "total_node_count": len(model.nodes) + len(extra_nodes),
        "c3d6_element_count": len(model.elements),
        "cin3d8_element_count": (
            len(left_inf["elements"]) + len(right_inf["elements"])
        ),
        "total_element_count": (
            len(model.elements)
            + len(left_inf["elements"])
            + len(right_inf["elements"])
        ),
        "finite_pbc_pair_count": len(finite_pairs),
        "infinite_far_pbc_pair_count": len(pairs) - len(finite_pairs),
        "total_pbc_pair_count": len(pairs),
        "x_coordinates_mm": {
            "left_infinite_interface": x_left_interface,
            "pnc_start": x_pnc_start,
            "excitation": x_excitation,
            "response": x_response,
            "right_infinite_interface": x_right_interface,
        },
        "nsets": {name: len(ids) for name, ids in nsets.items()},
        "elsets": {
            name: len(ids)
            for name, ids in model.elsets.items()
            if name
            in {
                "TRANSITION",
                "TRANSITION_LEFT",
                "TRANSITION_RIGHT",
                "PNC_ALL",
                "MONITOR_LEFT",
                "MONITOR_RIGHT",
            }
        },
        "surface_face_counts": {
            name: len(entries) for name, entries in surfaces.items()
        },
        "limitations": [
            "CIN3D8 is a structural radiation infinite element, not a PML.",
            (
                "Rectangular mode uses parallel auxiliary-node rays and is "
                "oriented toward plane-wave propagation."
                if args.infinite_geometry == "rectangular"
                else "Radial finite-pole mapping flares the displayed layer."
            ),
            "Quantitative use requires a uniform-strip reflection benchmark.",
            "Structural infinite elements are input-file driven and have limited Abaqus/CAE support.",
        ],
    }

    return model, cell, {
        "pairs": pairs,
        "finite_pairs": finite_pairs,
        "nsets": nsets,
        "surfaces": surfaces,
        "extra_nodes": extra_nodes,
        "left_inf": left_inf,
        "right_inf": right_inf,
        "meta": metadata,
    }

def write_deck(
    path: Path,
    model: ModelBuilder,
    cell: UnitCell,
    aux: dict,
    args,
) -> None:
    pairs: List[Tuple[int, int]] = aux["pairs"]
    nsets: Dict[str, List[int]] = aux["nsets"]
    surfaces: Dict[str, List[Tuple[int, str]]] = aux["surfaces"]
    extra_nodes: Dict[int, Vec3] = aux["extra_nodes"]
    left_inf = aux["left_inf"]
    right_inf = aux["right_inf"]
    meta = aux["meta"]

    base_alpha, base_beta = rayleigh_coefficients(
        args.base_zeta,
        args.damping_freq_min,
        args.damping_freq_max,
    )

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("*HEADING\n")
        f.write(
            "Fig.8(a) finite square-lattice PnC with CIN3D8 infinite elements\n"
        )
        f.write("** Units: mm, N, tonne, s; frequency in Hz\n")
        f.write(f"** Unit mesh: {Path(args.unit_inp).name}\n")
        f.write(f"** Repeated cells: {args.repeats}; a={cell.a:.12g} mm\n")
        f.write(
            "** Layout: left CIN3D8 | left transition | PnC | "
            "right transition | right CIN3D8\n"
        )
        f.write(
            f"** Transition target/max dx={args.length:.12g} mm; "
            f"actual dx={meta['actual_transition_x_length_mm']:.12g} mm\n"
        )
        f.write(
            f"** Infinite geometry={meta['infinite_geometry']}; "
            f"display length={meta['infinite_display_length_mm']:.12g} mm\n"
        )
        if meta["infinite_pole_distance_mm"] is not None:
            f.write(
                f"** Radial pole distance="
                f"{meta['infinite_pole_distance_mm']:.12g} mm\n"
            )
        f.write(
            f"** Excitation plane mode={args.excitation_plane}; "
            f"x={meta['excitation_x_mm']:.12g} mm; "
            f"{meta['excitation_plane_description']}\n"
        )
        f.write(
            f"** Input displacement amplitudes (mm): "
            f"Ux={args.ux:.12g}, Uy={args.uy:.12g}, Uz={args.uz:.12g}\n"
        )
        f.write("**\n")
        f.write("** CIN3D8 LIMITATION NOTE:\n")
        f.write(
            "** Structural infinite elements model an unbounded radiating exterior.\n"
        )
        f.write(
            "** Rectangular mode uses parallel rays; radial mode uses a finite pole.\n"
        )
        f.write(
            "** Validate reflection using a uniform strip before quantitative use.\n"
        )
        f.write("*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO\n")

        f.write("*NODE\n")
        for nid in sorted(model.nodes):
            x, y, z = model.nodes[nid]
            f.write(f"{nid}, {x:.12g}, {y:.12g}, {z:.12g}\n")
        for nid in sorted(extra_nodes):
            x, y, z = extra_nodes[nid]
            f.write(f"{nid}, {x:.12g}, {y:.12g}, {z:.12g}\n")

        f.write("*ELEMENT, TYPE=C3D6, ELSET=ALL_C3D6\n")
        for eid in sorted(model.elements):
            conn = model.elements[eid]
            f.write(f"{eid}, " + ", ".join(str(v) for v in conn) + "\n")

        f.write("*ELEMENT, TYPE=CIN3D8, ELSET=INFINITE_LEFT\n")
        for eid in sorted(left_inf["elements"]):
            conn = left_inf["elements"][eid]
            f.write(f"{eid}, " + ", ".join(str(v) for v in conn) + "\n")

        f.write("*ELEMENT, TYPE=CIN3D8, ELSET=INFINITE_RIGHT\n")
        for eid in sorted(right_inf["elements"]):
            conn = right_inf["elements"][eid]
            f.write(f"{eid}, " + ", ".join(str(v) for v in conn) + "\n")

        write_elset(
            f,
            "INFINITE_ALL",
            sorted(left_inf["elements"]) + sorted(right_inf["elements"]),
        )

        preferred = [
            "PNC_ALL",
            "TRANSITION",
            "TRANSITION_LEFT",
            "TRANSITION_RIGHT",
            "MONITOR_LEFT",
            "MONITOR_RIGHT",
        ]
        written = set()
        for name in preferred:
            if model.elsets.get(name):
                write_elset(f, name, model.elsets[name])
                written.add(name)
        for name in sorted(model.elsets):
            if name in written:
                continue
            if (
                name.startswith("PNC_CELL_")
                or name.startswith("TRANSITION_LEFT_X_")
                or name.startswith("TRANSITION_RIGHT_X_")
            ):
                write_elset(f, name, model.elsets[name])

        for name, ids in nsets.items():
            write_nset(f, name, ids)

        for name, entries in surfaces.items():
            f.write(f"*SURFACE, NAME={name}, TYPE=ELEMENT\n")
            for eid, face_name in entries:
                f.write(f"{eid}, {face_name}\n")

        # Match the official Abaqus CIN3D8 verification pattern: the infinite
        # surface is listed first and the regular finite-element surface second.
        f.write("*TIE, NAME=TIE_LEFT_INFINITY, ADJUST=NO\n")
        f.write("LEFT_INFINITY_INTERFACE, LEFT_TRANSITION_OUTER\n")
        f.write("*TIE, NAME=TIE_RIGHT_INFINITY, ADJUST=NO\n")
        f.write("RIGHT_INFINITY_INTERFACE, RIGHT_TRANSITION_OUTER\n")

        f.write(
            "** Zero-phase periodic equations: u_top - u_bottom = 0\n"
        )
        for top, bottom in pairs:
            for dof in (1, 2, 3):
                f.write("*EQUATION\n")
                f.write("2\n")
                f.write(
                    f"{top}, {dof}, 1.0, {bottom}, {dof}, -1.0\n"
                )

        f.write("*MATERIAL, NAME=ALUMINUM\n")
        f.write("*ELASTIC\n")
        f.write(
            f"{args.young_modulus:.12g}, {args.poisson_ratio:.12g}\n"
        )
        f.write("*DENSITY\n")
        f.write(f"{args.density:.12g}\n")
        if base_alpha > 0.0 or base_beta > 0.0:
            f.write(
                f"*DAMPING, ALPHA={base_alpha:.12g}, "
                f"BETA={base_beta:.12g}\n"
            )

        f.write("*SOLID SECTION, ELSET=PNC_ALL, MATERIAL=ALUMINUM\n")
        f.write(",\n")
        f.write("*SOLID SECTION, ELSET=TRANSITION, MATERIAL=ALUMINUM\n")
        f.write(",\n")
        f.write("*SOLID SECTION, ELSET=INFINITE_LEFT, MATERIAL=ALUMINUM\n")
        f.write(",\n")
        f.write("*SOLID SECTION, ELSET=INFINITE_RIGHT, MATERIAL=ALUMINUM\n")
        f.write(",\n")

        f.write("*STEP, NAME=STEADY_STATE_TRANSMISSION, PERTURBATION\n")
        f.write(
            "*STEADY STATE DYNAMICS, DIRECT, INTERVAL=RANGE, "
            "FREQUENCY SCALE=LINEAR\n"
        )
        if args.frequency_spec["mode"] == "single":
            f.write(
                f"{args.frequency_spec['frequency_hz']:.12g}, 0.\n"
            )
        else:
            f.write(
                f"{args.frequency_spec['lower_hz']:.12g}, "
                f"{args.frequency_spec['upper_hz']:.12g}, "
                f"{args.frequency_spec['points']}, 1.0\n"
            )
        f.write(
            "** Component-wise prescribed displacement at INPUT_DRIVEN.\n"
        )
        f.write("*BOUNDARY, REAL\n")
        for dof, amplitude in ((1, args.ux), (2, args.uy), (3, args.uz)):
            f.write(
                f"INPUT_DRIVEN, {dof}, {dof}, {amplitude:.12g}\n"
            )

        f.write("*OUTPUT, FIELD\n")
        f.write("*NODE OUTPUT\n")
        f.write("U\n")
        f.write("*ELEMENT OUTPUT, ELSET=MONITOR_LEFT\n")
        f.write("SENER, IVOL\n")
        f.write("*ELEMENT OUTPUT, ELSET=MONITOR_RIGHT\n")
        f.write("SENER, IVOL\n")
        f.write("*OUTPUT, HISTORY\n")
        f.write("*ENERGY OUTPUT\n")
        f.write("ALLSE, ALLKE, ALLVD\n")
        f.write("*END STEP\n")

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate the Abaqus Fig. 8(a) finite square-lattice PnC model "
            "terminated by CIN3D8 structural infinite elements."
        )
    )
    p.add_argument(
        "unit_inp",
        help="Unit-cell mesh .inp file containing C3D6 nodes/elements.",
    )
    p.add_argument(
        "repeats",
        type=int,
        help="Number of unit cells repeated along x.",
    )
    p.add_argument(
        "length",
        type=float,
        nargs="?",
        default=2.0,
        help=(
            "Maximum x-direction C3D6 element length in each transition "
            "region, in mm; default 2.0."
        ),
    )
    p.add_argument("-o", "--output", help="Output Abaqus .inp file.")
    p.add_argument(
        "--node-set",
        help="Source *NODE NSET name; auto-detected by default.",
    )
    p.add_argument(
        "--element-set",
        help="Source C3D6 ELSET name; auto-detected by default.",
    )
    p.add_argument(
        "--transition-length-cells",
        type=float,
        default=2.0,
        help="Length of each left/right transition divided by a; default 2.0.",
    )
    p.add_argument(
        "--infinite-geometry",
        choices=("rectangular", "radial"),
        default="rectangular",
        help=(
            "CIN3D8 auxiliary-node geometry. 'rectangular' keeps the outer "
            "face the same y-z size by translating nodes only along x; "
            "'radial' uses a finite pole and flares outward. Default rectangular."
        ),
    )
    p.add_argument(
        "--infinite-display-length-cells",
        type=float,
        default=2.0,
        help=(
            "Displayed distance from the CIN3D8 terminating face to its "
            "auxiliary far face, divided by a; default 2.0."
        ),
    )
    p.add_argument(
        "--infinite-pole-distance-cells",
        type=float,
        default=2.0,
        help=(
            "Radial-mode distance from each CIN3D8 terminating surface to its "
            "pole, divided by a; default 2.0. Ignored in rectangular mode."
        ),
    )
    p.add_argument(
        "--no-infinite-far-pbc",
        action="store_true",
        help=(
            "Do not impose top-bottom periodic equations on the CIN3D8 far "
            "interpolation nodes."
        ),
    )
    p.add_argument(
        "--merge-tolerance",
        type=float,
        default=1.0e-7,
        help="Coordinate tolerance for finite-mesh node merging.",
    )
    p.add_argument(
        "--base-zeta",
        type=float,
        default=0.001,
        help="Small target material damping ratio in the analysis band.",
    )
    p.add_argument(
        "--freq",
        type=float,
        default=None,
        help=(
            "Single calculation frequency in Hz. When this option is used, "
            "do not also provide --freq-min, --freq-max, or --freq-points."
        ),
    )
    p.add_argument(
        "--freq-min",
        type=float,
        default=None,
        help=(
            "Sweep lower frequency in Hz. Use together with --freq-max "
            "and --freq-points."
        ),
    )
    p.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help=(
            "Sweep upper frequency in Hz. Use together with --freq-min "
            "and --freq-points."
        ),
    )
    p.add_argument(
        "--freq-points",
        type=int,
        default=None,
        help=(
            "Number of sweep points. Use together with --freq-min "
            "and --freq-max."
        ),
    )
    p.add_argument(
        "--excitation-plane",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "0 = PnC entrance / left-transition interface; "
            "1 = left-CIN3D8 / left-transition interface; default 1."
        ),
    )
    p.add_argument(
        "--ux",
        type=float,
        default=2.0e-5,
        help="Real Ux input amplitude in mm; default 2e-5 mm.",
    )
    p.add_argument(
        "--uy",
        type=float,
        default=2.0e-5,
        help="Real Uy input amplitude in mm; default 2e-5 mm.",
    )
    p.add_argument(
        "--uz",
        type=float,
        default=0.0,
        help="Real Uz input amplitude in mm; default 0.",
    )
    p.add_argument(
        "--young-modulus",
        type=float,
        default=70000.0,
        help="Young's modulus in N/mm^2; default 70000.",
    )
    p.add_argument("--poisson-ratio", type=float, default=0.33)
    p.add_argument(
        "--density",
        type=float,
        default=2.6314e-9,
        help="Density in tonne/mm^3; default equals 2631.4 kg/m^3.",
    )
    return p


def check_args(args) -> None:
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if args.length <= 0.0:
        raise ValueError("length must be positive.")
    if args.transition_length_cells <= 0.0:
        raise ValueError("--transition-length-cells must be positive.")
    if args.infinite_display_length_cells <= 0.0:
        raise ValueError("--infinite-display-length-cells must be positive.")
    if args.infinite_pole_distance_cells <= 0.0:
        raise ValueError("--infinite-pole-distance-cells must be positive.")
    if not (0.0 <= args.poisson_ratio < 0.5):
        raise ValueError("Poisson's ratio must satisfy 0 <= nu < 0.5.")
    if args.young_modulus <= 0.0 or args.density <= 0.0:
        raise ValueError("Young's modulus and density must be positive.")
    range_values = (
        args.freq_min,
        args.freq_max,
        args.freq_points,
    )
    range_was_supplied = tuple(value is not None for value in range_values)

    if args.freq is not None:
        if any(range_was_supplied):
            raise ValueError(
                "--freq cannot be combined with --freq-min, "
                "--freq-max, or --freq-points."
            )
        if args.freq <= 0.0:
            raise ValueError("--freq must be positive.")

        args.frequency_spec = {
            "mode": "single",
            "frequency_hz": args.freq,
        }
        args.damping_freq_min = args.freq
        args.damping_freq_max = args.freq
    else:
        if any(range_was_supplied) and not all(range_was_supplied):
            raise ValueError(
                "A frequency range requires --freq-min, --freq-max, "
                "and --freq-points together."
            )

        if not any(range_was_supplied):
            args.freq_min = 40000.0
            args.freq_max = 70000.0
            args.freq_points = 151

        if args.freq_min <= 0.0 or args.freq_max <= args.freq_min:
            raise ValueError("Require 0 < freq-min < freq-max.")
        if args.freq_points < 2:
            raise ValueError("--freq-points must be at least 2.")

        args.frequency_spec = {
            "mode": "range",
            "lower_hz": args.freq_min,
            "upper_hz": args.freq_max,
            "points": args.freq_points,
        }
        args.damping_freq_min = args.freq_min
        args.damping_freq_max = args.freq_max
    if args.excitation_plane not in (0, 1):
        raise ValueError("--excitation-plane must be 0 or 1.")
    if args.base_zeta < 0.0:
        raise ValueError("--base-zeta must be nonnegative.")


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        check_args(args)
        output = (
            Path(args.output)
            if args.output
            else Path(args.unit_inp)
            .with_suffix("")
            .with_name(
                Path(args.unit_inp).stem
                + f"_fig8a_CIN3D8_N{args.repeats}.inp"
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)

        model, cell, aux = build_model(args)
        write_deck(output, model, cell, aux, args)
        manifest = output.with_suffix(".json")
        manifest_data = dict(aux["meta"])
        manifest_data["steady_frequency"] = args.frequency_spec
        manifest.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Wrote Abaqus deck : {output}")
        print(f"Wrote manifest    : {manifest}")
        print(
            f"Nodes             : {aux['meta']['total_node_count']} "
            f"({aux['meta']['finite_node_count']} finite + "
            f"{aux['meta']['infinite_auxiliary_node_count']} CIN auxiliary)"
        )
        print(
            f"Elements          : {aux['meta']['total_element_count']} "
            f"({aux['meta']['c3d6_element_count']} C3D6 + "
            f"{aux['meta']['cin3d8_element_count']} CIN3D8)"
        )
        print(
            f"PBC node pairs    : {aux['meta']['total_pbc_pair_count']}"
        )
        print(
            f"Excitation plane  : mode {args.excitation_plane}, "
            f"x={aux['meta']['excitation_x_mm']:.12g} mm"
        )
        print(
            f"Infinite geometry  : {args.infinite_geometry}; "
            f"display length={aux['meta']['infinite_display_length_mm']:.12g} mm"
        )
        print(
            "Infinite boundary  : two tied CIN3D8 layers; "
            "uniform-strip reflection validation required."
        )
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
