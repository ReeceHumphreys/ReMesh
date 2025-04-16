import numpy as np
import trimesh


def load_mesh(path):
    """Load the mesh from the specified path."""
    return trimesh.load(path)


def save_mesh(mesh, path):
    """Save the transformed mesh to the specified path."""
    mesh.export(path)
    print(f"Transformed mesh saved to {path}")


def apply_transformations(mesh, transformations):
    """Apply transformations from the config file."""
    _apply_rotation(mesh, transformations)
    _apply_axis_flip(mesh, transformations)
    _apply_scaling(mesh, transformations)
    _apply_translation(mesh, transformations)
    return mesh


def _apply_axis_flip(mesh, transformations):
    """Apply axis flipping based on the config file."""
    if "flip_axes" in transformations:
        flip_config = transformations["flip_axes"]
        flip_matrix = np.eye(4)

        if flip_config.get("x", False):
            flip_matrix[0, 0] = -1
        if flip_config.get("y", False):
            flip_matrix[1, 1] = -1
        if flip_config.get("z", False):
            flip_matrix[2, 2] = -1

        mesh.apply_transform(flip_matrix)
        print(f"Flipped axes: {flip_config}")


def _apply_translation(mesh, transformations):
    """Apply a user-defined translation vector to the mesh."""
    if "translate" in transformations:
        translation_vector = np.array(transformations["translate"])
        mesh.apply_translation(translation_vector)
        print(f"Translated model by {translation_vector}.")


def _apply_rotation(mesh, transformations):
    """Apply rotation based on the config file."""
    if "rotate" in transformations:
        angles = np.radians(transformations["rotate"])
        rotation_matrix = trimesh.transformations.euler_matrix(*angles)
        mesh.apply_transform(rotation_matrix)
        print(f"Rotated model by {transformations['rotate']} degrees.")


def _apply_scaling(mesh, transformations):
    """Apply scaling based on the config file."""
    if "desired_dimensions" in transformations:
        desired_length = transformations["desired_dimensions"].get("length")
        desired_width = transformations["desired_dimensions"].get("width")
        desired_height = transformations["desired_dimensions"].get("height")

        # Get current model dimensions
        current_length, current_width, current_height = mesh.extents

        # Compute scale factors only for provided dimensions
        scale_factors = []
        if desired_length:
            scale_factors.append(desired_length / current_length)
        if desired_width:
            scale_factors.append(desired_width / current_width)
        if desired_height:
            scale_factors.append(desired_height / current_height)

        if not scale_factors:
            print("No valid scale factors provided. Skipping scaling.")
            return

        if transformations["desired_dimensions"].get("keep_aspect_ratio", True):
            # Use the smallest scale factor to maintain proportions
            scale_factor = min(scale_factors)
            print(f"Uniformly scaling model by factor {scale_factor}.")
            mesh.apply_scale(scale_factor)
        else:
            # Independent scaling for X, Y, and Z (default to 1.0 if not specified)
            scale_factor_length = (
                desired_length / current_length if desired_length else 1.0
            )
            scale_factor_width = desired_width / current_width if desired_width else 1.0
            scale_factor_height = (
                desired_height / current_height if desired_height else 1.0
            )

            print(
                f"Scaling model to {desired_length or current_length}m x "
                f"{desired_width or current_width}m x {desired_height or current_height}m."
            )

            mesh.apply_scale(
                [scale_factor_length, scale_factor_width, scale_factor_height]
            )


# def main():
#     parser = argparse.ArgumentParser(description="Transform a 3D STL model.")
#     parser.add_argument(
#         "config", type=str, help="Path to the transformation config file."
#     )
#     parser.add_argument("input_stl", type=str, help="Path to the input STL file.")
#     parser.add_argument(
#         "output_stl", type=str, help="Path to save the transformed STL file."
#     )

#     args = parser.parse_args()

#     # Load configuration
#     with open(args.config, "r") as file:
#         transformations = yaml.safe_load(file)

#     # Load and transform the mesh
#     mesh = load_mesh(args.input_stl)
#     apply_transformations(mesh, transformations)

#     # Save the transformed mesh
#     save_mesh(mesh, args.output_stl)
