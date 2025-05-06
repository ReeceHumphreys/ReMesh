import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class __Bounds:
    """
    Axis-aligned bounding box defined by min/max for x, y, and z.

    Attributes:
        x: Tuple of (min, max) along the X axis.
        y: Tuple of (min, max) along the Y axis.
        z: Tuple of (min, max) along the Z axis.
    """

    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]

    def validate(self) -> None:
        """
        Ensure that for each axis, the minimum bound is not greater than the maximum.
        """
        for axis in ("x", "y", "z"):
            lo, hi = getattr(self, axis)
            if lo > hi:
                raise ValueError(f"Bounds for '{axis}' are invalid: {lo} > {hi}")


@dataclass
class Region:
    """
    Configuration for a mesh refinement region.

    Attributes:
        name: Identifier for the region.
        bounds: Axis-aligned bounds of the region.
        operation: 'subdivide' or 'combine'.
        passes: Number of iterations (0 = until stable).
        max_edge_length: Threshold for subdivision edges.
        reduction_ratio: Fraction retained for combining.
        use_spherical_interp: Whether to project new points to a sphere.
        curvature_center: Center of sphere for projection, if used.
    """

    name: str
    bounds: __Bounds
    operation: str = "subdivide"  # 'subdivide' or 'combine'
    passes: int = 1  # 0 = until stable
    max_edge_length: Optional[float] = None
    reduction_ratio: Optional[float] = None
    use_spherical_interp: bool = False
    curvature_center: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """
        Validate region parameters after initialization.
        """
        self.bounds.validate()
        if self.operation not in ("subdivide", "combine"):
            raise ValueError(
                f"Region '{self.name}': invalid operation '{self.operation}'"
            )
        if self.passes < 0:
            raise ValueError(f"Region '{self.name}': passes must be non-negative")
        if self.operation == "subdivide" and self.max_edge_length is None:
            raise ValueError(
                f"Region '{self.name}': max_edge_length is required for subdivision"
            )
        if self.operation == "combine" and self.reduction_ratio is None:
            raise ValueError(
                f"Region '{self.name}': reduction_ratio is required for combining"
            )


def load_regions(config_path: Path) -> List[Region]:
    """
    Load and validate region configurations from a YAML file.

    Parameters:
        config_path: Path to the YAML config file containing a 'regions' list.

    Returns:
        A list of Region instances.

    Raises:
        ValueError: If the YAML is missing the 'regions' key or contains invalid data.
    """
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict) or "regions" not in raw:
        raise ValueError("Config must contain a top-level 'regions' key")

    regions: List[Region] = []
    for entry in raw["regions"]:
        regions.append(
            Region(
                name=entry["name"],
                bounds=__Bounds(**entry["bounds"]),
                operation=entry.get("operation", "subdivide"),
                passes=entry.get("passes", 1),
                max_edge_length=entry.get("max_edge_length"),
                reduction_ratio=entry.get("reduction_ratio"),
                use_spherical_interp=entry.get("use_spherical_interp", False),
                curvature_center=entry.get("curvature_center"),
            )
        )
    return regions
