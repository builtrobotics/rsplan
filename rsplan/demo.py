from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np

import rsplan

# List of path end poses to visualize Reeds-Shepp paths for with matplotlib.
# Format: (end x, end y, end yaw, turn radius, runway length)
_END_POSES: List[Tuple[float, float, float, int, float]] = [
    (5, 6, np.pi, 1, 0),
    (15, 3, np.pi / 2.0, 2, 6),
    (-2, -4, np.pi, 4, 3),
    (-7, 2, np.pi, 4, 0),
    (-7, -7, 0.0, 6, 1),
    (0.7, 1.8, 1, 1, 1),
    (-5, 6, np.pi / 3.0, 2, 1),
    (7, 2, 0.0, 6, 3),
    (-4, -1, -np.pi / 2.0, 1, 3),
    (-4, -1, -np.pi / 2.0, 2, 3),
    (-4, -1, -np.pi / 2.0, 4, 3),
    (-4, -1, -np.pi / 2.0, 6, 3),
    (1.41513, 5.670786, 1.08317, 1, 3),
]


def _plot_arrow(
    x: float,
    y: float,
    yaw: float,
    length: float = 0.3,
    width: float = 0.2,
    label: str = "",
) -> None:
    """Adds an arrow to the plot."""
    plt.arrow(
        x,
        y,
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=width,
        head_length=width,
    )
    plt.plot(x, y, marker="s", label=label)


def _viz_path(rs_path: rsplan.Path, path_num: int) -> None:
    """Visualizes the given path in the plot."""
    x_coords, y_coords, _ = rs_path.coordinates_tuple()

    path_plt = plt.subplot(111)
    path_plt.plot(x_coords, y_coords, label=(f"Path ix: {path_num}"))

    # Shrink current axis's height by 10% on the bottom
    box = path_plt.get_position()
    path_plt.set_position(
        [box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95]
    )

    # Put a legend below current axis
    path_plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=3
    )
    plt.show(block=False)


def _demo_scene() -> None:
    """Generate and visualize the set of pre-defined paths solved by the path planner."""
    plt.cla()
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    start = (0.0, 0.0, 0.0)

    for i in range(len(_END_POSES)):
        end_coords = _END_POSES[i]
        x, y, yaw, turn_radius, runway_length = end_coords
        step_size = 0.05

        _plot_arrow(*start)  # Start arrow same for all paths starting at origin

        # Passing in yaw angles in radians
        rs_path = rsplan.path(
            start, (x, y, yaw), turn_radius, runway_length, step_size
        )
        _viz_path(rs_path, i + 1)
        _plot_arrow(x, y, yaw)  # End of path arrow to show direction

    plt.show()


if __name__ == "__main__":
    _demo_scene()
