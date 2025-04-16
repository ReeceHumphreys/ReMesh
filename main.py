import argparse
import yaml
from transform import apply_transformations
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_mesh(path):
    return trimesh.load_mesh(path, process=False)


def triangle_centroid(tri):
    return np.mean(tri, axis=0)


def needs_subdivision(triangle, max_len):
    def triangle_edge_lengths(triangle):
        return [np.linalg.norm(triangle[i] - triangle[(i + 1) % 3]) for i in range(3)]

    return any(e > max_len for e in triangle_edge_lengths(triangle))


def is_in_bounds(centroid, bounds):
    x, y, z = centroid
    return (
        bounds["x"][0] <= x <= bounds["x"][1]
        and bounds["y"][0] <= y <= bounds["y"][1]
        and bounds["z"][0] <= z <= bounds["z"][1]
    )


def tag_triangles_by_region(mesh, regions):
    tagged = []
    for i, face in enumerate(mesh.faces):
        tri = mesh.vertices[face]
        centroid = triangle_centroid(tri)
        for region in regions:
            if is_in_bounds(centroid, region["bounds"]):
                tagged.append((i, region))
                break
    return tagged


def midpoint(p1, p2):
    return (p1 + p2) / 2


def project_to_sphere(p, center):
    v = p - center
    radius = np.linalg.norm(v)
    return center + v / np.linalg.norm(v) * radius


def subdivide_triangle(vertices, face, curvature_center=None):
    v0, v1, v2 = vertices[face]
    a = midpoint(v0, v1)
    b = midpoint(v1, v2)
    c = midpoint(v2, v0)

    if curvature_center is not None:
        a = project_to_sphere(a, curvature_center)
        b = project_to_sphere(b, curvature_center)
        c = project_to_sphere(c, curvature_center)

    base_index = len(vertices)
    vertices = np.vstack([vertices, a, b, c])
    ia, ib, ic = base_index, base_index + 1, base_index + 2
    i0, i1, i2 = face

    new_faces = np.array([[i0, ia, ic], [ia, i1, ib], [ib, i2, ic], [ia, ib, ic]])
    return vertices, new_faces


def combine_faces(mesh, face_indices, reduction_ratio):
    # Extract region submesh
    submesh = mesh.submesh([face_indices], append=True, repair=False)
    # Determine target face count
    target_count = max(1, int(len(submesh.faces) * reduction_ratio))
    # Simplify
    simplified = submesh.simplify_quadratic_decimation(target_count)
    return simplified


def refine_mesh(mesh, tagged_faces):
    vertices = mesh.vertices.copy()
    faces_list = []
    processed = set()

    # Group faces by region
    region_groups = {}
    for face_idx, region in tagged_faces:
        key = region["name"]
        region_groups.setdefault(key, {"region": region, "faces": []})
        region_groups[key]["faces"].append(face_idx)

    # Process each region
    for key, data in region_groups.items():
        region = data["region"]
        face_indices = data["faces"]
        op = region.get("operation", "subdivide")

        if op == "subdivide":
            for face_idx in face_indices:
                face = mesh.faces[face_idx]
                tri = vertices[face]
                if needs_subdivision(tri, region["max_edge_length"]):
                    curvature_center = (
                        region.get("curvature_center")
                        if region.get("use_spherical_interp")
                        else None
                    )
                    vertices, new_faces = subdivide_triangle(
                        vertices, face, curvature_center
                    )
                    faces_list.extend(new_faces)
                else:
                    faces_list.append(face)
                processed.add(face_idx)

        elif op == "combine":
            # Combine via decimation
            simplified = combine_faces(mesh, face_indices, region["reduction_ratio"])
            # Append simplified mesh, adjusting indices
            offset = len(vertices)
            vertices = np.vstack([vertices, simplified.vertices])
            new_faces = simplified.faces + offset
            for f in new_faces:
                faces_list.append(f)
            processed.update(face_indices)

        else:
            # No-op: keep original faces
            for face_idx in face_indices:
                faces_list.append(mesh.faces[face_idx])
                processed.add(face_idx)

    # Add untouched faces
    for i, face in enumerate(mesh.faces):
        if i not in processed:
            faces_list.append(face)

    all_faces = np.array(faces_list)
    return trimesh.Trimesh(vertices=vertices, faces=all_faces, process=False)


def show_before_after_colored(
    before, after, tagged_before, tagged_after, title="Before vs After"
):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax1.set_title("Before Refinement")
    ax2.set_title("After Refinement")

    def color_faces(ax, mesh, tagged):
        region_colors = {}
        region_faces = {}

        for face_idx, region in tagged:
            name = region["name"]
            if name not in region_colors:
                region_colors[name] = np.random.rand(
                    3,
                )
                region_faces[name] = []
            region_faces[name].append(mesh.faces[face_idx])

        for name, faces in region_faces.items():
            tri_verts = mesh.vertices[np.array(faces)]
            poly = Poly3DCollection(
                tri_verts, facecolor=region_colors[name], edgecolor="k", alpha=0.6
            )
            ax.add_collection3d(poly)

        ax.auto_scale_xyz(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2])

    color_faces(ax1, before, tagged_before)
    color_faces(ax2, after, tagged_after)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to refinement config YAML")
    parser.add_argument("--out", default="refined.stl", help="Output STL path")
    parser.add_argument(
        "--debug", action="store_true", help="Show region debug visuals"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    mesh = load_mesh(config["model"]["input_stl"])
    mesh = apply_transformations(mesh, config["model"]["transformations"])

    regions = config.get("regions", [])
    if not regions:
        print("No regions defined for refinement.")
        return

    tagged = tag_triangles_by_region(mesh, regions)
    refined = refine_mesh(mesh, tagged)

    if args.debug:
        tagged_after = tag_triangles_by_region(refined, regions)
        show_before_after_colored(mesh, refined, tagged, tagged_after)

    refined.export(args.out)
    print(f"Refined mesh written to {args.out}")


if __name__ == "__main__":
    main()
