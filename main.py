#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import List
from regions import Region, load_regions
import numpy as np
import trimesh
import logging


def triangle_edge_lengths(tri: np.ndarray) -> List[float]:
    """Compute the three edge lengths of a triangle."""
    return [np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3)]


def needs_subdivision(tri: np.ndarray, max_len: float) -> bool:
    """True if any edge of `tri` exceeds `max_len`."""
    return any(l > max_len for l in triangle_edge_lengths(tri))


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


def rebuild_mesh(
    old: trimesh.Trimesh, verts: np.ndarray, faces: np.ndarray
) -> trimesh.Trimesh:
    """Construct a new Trimesh with process=False to preserve metadata."""
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def get_region_faces(mesh: trimesh.Trimesh, region: Region) -> np.ndarray:
    """
    Return array of face indices whose centroid lies within region.bounds,
    computed in a fully vectorized manner.
    """
    # shape (F,3,3) → (F,3,3)
    face_verts = mesh.vertices[mesh.faces]
    # compute centroids shape (F,3)
    cents = face_verts.mean(axis=1)

    bx, by, bz = region.bounds.x, region.bounds.y, region.bounds.z
    mask = (
        (cents[:, 0] >= bx[0])
        & (cents[:, 0] <= bx[1])
        & (cents[:, 1] >= by[0])
        & (cents[:, 1] <= by[1])
        & (cents[:, 2] >= bz[0])
        & (cents[:, 2] <= bz[1])
    )
    return np.nonzero(mask)[0]


def process_region(mesh: trimesh.Trimesh, region: Region) -> trimesh.Trimesh:
    # Set up logger and initial parameters
    name, op, passes = region.name, region.operation, region.passes
    logger = logging.getLogger(name)
    logger.info(f"Plan: {passes or '∞'} passes, operation={op}")

    refined = mesh
    itr = 0

    # Loop until completion or stable mesh
    while True:
        itr += 1
        # Identify faces within the region bounds for this iteration
        fids = get_region_faces(refined, region)
        if len(fids) == 0:
            logger.info("no faces in region; stopping early")
            break

        verts, faces_out = refined.vertices.copy(), []
        used = set()

        # Process subdivision operation
        if op == "subdivide":
            center = (
                np.array(region.curvature_center)
                if region.use_spherical_interp
                else None
            )
            for fid in fids:
                tri = verts[refined.faces[fid]]
                if needs_subdivision(tri, region.max_edge_length):
                    # Compute midpoints of the triangle edges
                    a, b, c = (
                        (tri[0] + tri[1]) / 2,
                        (tri[1] + tri[2]) / 2,
                        (tri[2] + tri[0]) / 2,
                    )
                    # If spherical interpolation is enabled, project midpoints onto the sphere
                    if center is not None:
                        # project midpoints to sphere
                        for pt in (a, b, c):
                            pt[:] = center + (pt - center) / np.linalg.norm(
                                pt - center
                            ) * np.linalg.norm(pt - center)
                    start = len(verts)
                    verts = np.vstack([verts, a, b, c])
                    ia, ib, ic = start, start + 1, start + 2
                    i0, i1, i2 = refined.faces[fid]
                    new_quads = np.array(
                        [[i0, ia, ic], [ia, i1, ib], [ib, i2, ic], [ia, ib, ic]]
                    )
                    faces_out.extend(new_quads)
                else:
                    faces_out.append(refined.faces[fid])
                used.add(fid)

        # Process combine (decimation) operation
        elif op == "combine":
            simp = combine_faces(refined, list(fids), region.reduction_ratio)
            kept_mask = np.ones(len(refined.faces), bool)
            kept_mask[fids] = False
            kept = refined.faces[kept_mask]
            offset = len(verts)
            verts = np.vstack([verts, simp.vertices])
            faces_out = list(kept) + [f + offset for f in simp.faces]
            used.update(fids)

        # Merge refined faces with untouched faces
        # carry over untouched faces
        for i, face in enumerate(refined.faces):
            if i not in used:
                faces_out.append(face)

        # Rebuild the mesh with updated vertices and faces
        refined = rebuild_mesh(refined, verts, np.array(faces_out))
        logger.info(f"completed iteration {itr}")

        # Check whether we've reached the required number of passes or stable state
        # stopping criteria
        if passes > 0 and itr >= passes:
            logger.info(f"reached {passes} passes; done")
            break

        if passes == 0:
            new_fids = get_region_faces(refined, region)
            if set(new_fids) == set(fids):
                logger.info(f"stable at iteration {itr}; done")
                break

    return refined


def parse_cli():
    p = argparse.ArgumentParser("Region-based mesh refinement")
    p.add_argument("stl", type=Path, help="input STL file")
    p.add_argument("config", type=Path, help="YAML region config")
    p.add_argument("--out", type=Path, default=Path("refined.stl"))
    p.add_argument("--debug", action="store_true", help="enable debug plots")
    return p.parse_args()


def main():
    args = parse_cli()
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.WARNING)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
        handlers=[out_handler, err_handler],
    )

    mesh = trimesh.load_mesh(args.stl, process=False)

    regions = load_regions(args.config)
    refined = mesh.copy()

    for reg in regions:
        refined = process_region(refined, reg)

    if args.debug:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        before_tags = [(i, r) for r in regions for i in get_region_faces(mesh, r)]
        after_tags = [(i, r) for r in regions for i in get_region_faces(refined, r)]

        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 6))
        for ax, m, tags, title in zip(
            axes, (mesh, refined), (before_tags, after_tags), ("Before", "After")
        ):
            ax.set_title(title)
            per = {}
            for fid, r in tags:
                per.setdefault(r.name, []).append(m.faces[fid])
            for name, fs in per.items():
                verts = m.vertices[np.array(fs)]
                poly = Poly3DCollection(verts, alpha=0.6, edgecolor="k")
                ax.add_collection3d(poly)
            ax.auto_scale_xyz(*m.vertices.T)
        plt.tight_layout()
        plt.show()

    print(
        f"[STATS] faces_in={len(mesh.faces)} faces_out={len(refined.faces)}", flush=True
    )
    print(f"[STATUS] completed refinement: output={args.out}", flush=True)

    refined.export(str(args.out))
    logging.getLogger("main").info(f"exported refined mesh to {args.out!r}")


if __name__ == "__main__":
    main()
