#!/usr/bin/env python3
import argparse
import yaml
import trimesh
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Tuple, Optional, Any


def triangle_centroid(tri: np.ndarray) -> np.ndarray:
    """Return centroid of a 3×3 array of triangle vertices."""
    return tri.mean(axis=0)


def triangle_edge_lengths(tri: np.ndarray) -> List[float]:
    """Compute the three edge lengths of a triangle."""
    return [np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3)]


def needs_subdivision(tri: np.ndarray, max_len: float) -> bool:
    """True if any edge of `tri` exceeds `max_len`."""
    return any(length > max_len for length in triangle_edge_lengths(tri))


def in_bounds(point: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> bool:
    """Check if point lies within the axis-aligned box `bounds`."""
    x, y, z = point
    bx, by, bz = bounds["x"], bounds["y"], bounds["z"]
    return bx[0] <= x <= bx[1] and by[0] <= y <= by[1] and bz[0] <= z <= bz[1]


def tag_triangles_by_region(
    mesh: trimesh.Trimesh, regions: List[Dict[str, Any]]
) -> List[Tuple[int, Dict[str, Any]]]:
    """Return list of (face_index, region) for faces whose centroid lies in any region."""
    tags = []
    for idx, face in enumerate(mesh.faces):
        cent = triangle_centroid(mesh.vertices[face])
        for region in regions:
            if in_bounds(cent, region["bounds"]):
                tags.append((idx, region))
                break
    return tags


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return midpoint of two points."""
    return (a + b) / 2


def project_to_sphere(p: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Project point p outward radially from center onto sphere of radius |p-center|."""
    v = p - center
    return center + v / np.linalg.norm(v) * np.linalg.norm(v)


def subdivide_triangle(
    verts: np.ndarray, face: np.ndarray, curvature_center: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split one face into 4 smaller ones.
    Optionally project new points onto a sphere around curvature_center.
    """
    v0, v1, v2 = verts[face]
    a, b, c = midpoint(v0, v1), midpoint(v1, v2), midpoint(v2, v0)

    if curvature_center is not None:
        a, b, c = (project_to_sphere(pt, curvature_center) for pt in (a, b, c))

    start = len(verts)
    new_verts = np.vstack([verts, a, b, c])
    ia, ib, ic = start, start + 1, start + 2
    i0, i1, i2 = face

    new_faces = np.array(
        [
            [i0, ia, ic],
            [ia, i1, ib],
            [ib, i2, ic],
            [ia, ib, ic],
        ]
    )
    return new_verts, new_faces


def combine_faces(
    mesh: trimesh.Trimesh, face_indices: List[int], reduction_ratio: float
) -> trimesh.Trimesh:
    """
    Extract submesh by face_indices, decimate to
    `len(submesh.faces)*reduction_ratio` faces, and return it.
    """
    sub = mesh.submesh([face_indices], append=True, repair=False)
    target = max(1, int(len(sub.faces) * reduction_ratio))
    return sub.simplify_quadratic_decimation(target)


def refine_mesh(
    mesh: trimesh.Trimesh, tags: List[Tuple[int, Dict[str, Any]]]
) -> trimesh.Trimesh:
    """Apply per-region subdivision or decimation, return new Trimesh."""
    verts = mesh.vertices.copy()
    new_faces = []
    processed = set()

    # Group face indices by region
    groups: Dict[str, Dict[str, Any]] = {}
    for idx, region in tags:
        key = region["name"]
        groups.setdefault(key, {"region": region, "faces": []})["faces"].append(idx)

    for data in groups.values():
        region, face_ids = data["region"], data["faces"]
        op = region.get("operation", "subdivide")

        if op == "subdivide":
            for fid in face_ids:
                tri = verts[mesh.faces[fid]]
                if needs_subdivision(tri, region["max_edge_length"]):
                    center = (
                        np.array(region["curvature_center"])
                        if region.get("use_spherical_interp")
                        else None
                    )
                    verts, faces4 = subdivide_triangle(verts, mesh.faces[fid], center)
                    new_faces.extend(faces4)
                else:
                    new_faces.append(mesh.faces[fid])
                processed.add(fid)

        elif op == "combine":
            simp = combine_faces(mesh, face_ids, region["reduction_ratio"])
            offset = len(verts)
            verts = np.vstack([verts, simp.vertices])
            new_faces.extend(simp.faces + offset)
            processed.update(face_ids)

        else:  # no-op
            for fid in face_ids:
                new_faces.append(mesh.faces[fid])
                processed.add(fid)

    # add untouched faces
    for idx, face in enumerate(mesh.faces):
        if idx not in processed:
            new_faces.append(face)

    return trimesh.Trimesh(vertices=verts, faces=np.array(new_faces), process=False)


def show_before_after(
    before: trimesh.Trimesh,
    after: trimesh.Trimesh,
    tags_before: List[Tuple[int, Dict[str, Any]]],
    tags_after: List[Tuple[int, Dict[str, Any]]],
    title: str = "Mesh Refinement",
) -> None:
    """Visualize regions in different colors before/after refinement."""
    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    for ax, mesh, tags, t in zip(
        axes, (before, after), (tags_before, tags_after), ("Before", "After")
    ):
        ax.set_title(t)
        # collect faces per region
        faces_per = {}
        for fid, region in tags:
            faces_per.setdefault(region["name"], []).append(mesh.faces[fid])
        # plot each region
        for name, faces in faces_per.items():
            verts = mesh.vertices[np.array(faces)]
            poly = Poly3DCollection(
                verts, facecolor=np.random.rand(3), edgecolor="k", alpha=0.6
            )
            ax.add_collection3d(poly)
        ax.auto_scale_xyz(*mesh.vertices.T)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Region-based mesh refinement")
    p.add_argument("stl", type=Path, help="input STL path")
    p.add_argument("config", type=Path, help="regions YAML config")
    p.add_argument("--out", type=Path, default=Path("refined.stl"), help="output STL")
    p.add_argument("--debug", action="store_true", help="show before/after plot")
    return p.parse_args()


def main():
    args = parse_cli()
    mesh = trimesh.load_mesh(args.stl, process=False)
    cfg = yaml.safe_load(args.config.read_text())
    regions = cfg.get("regions", [])
    if not regions:
        print("No regions defined; exiting.")
        return

    tags_b = tag_triangles_by_region(mesh, regions)
    refined = refine_mesh(mesh, tags_b)

    if args.debug:
        tags_a = tag_triangles_by_region(refined, regions)
        show_before_after(mesh, refined, tags_b, tags_a)

    refined.export(str(args.out))
    print(f"Refined mesh saved to {args.out}")


if __name__ == "__main__":
    main()
