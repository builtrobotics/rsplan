from typing import Any, Literal, Tuple, Union

import numpy as np

########################################################################################
# GENERAL HELPER FUNCTIONS #############################################################
########################################################################################


def change_base(
    new_base: Tuple[float, float, float], end_pose2d: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Given new_base = (x1, y1, theta1) and end_pose2d = (x2, y2, theta2) represented
    in a coordinate system with origin (0, 0) and rotation 0 (in radians), return the
    position and rotation of end_pose2d in the coordinate system with origin (x1, y1)
    and rotation theta1.
    """
    dx = end_pose2d[0] - new_base[0]
    dy = end_pose2d[1] - new_base[1]
    xb, yb = rotate(dx, dy, -new_base[2])

    dtheta = end_pose2d[2] - new_base[2]

    return xb, yb, dtheta

    return *(helpers.rotate(end[0] - start[0], end[1] - start[1], -start[2])), (end[2] - start[2])


def rotate(
    x: Union[float, np.ndarray[Any, np.dtype[np.floating[Any]]]],
    y: Union[float, np.ndarray[Any, np.dtype[np.floating[Any]]]],
    psi: float,
) -> Tuple[Any, Any]:
    """Rotate all coordinates in the x and y lists counterclockwise by the angle psi (in
    radians). To rotate clockwise, pass in a negative angle.
    """
    s = np.sin(psi)
    c = np.cos(psi)

    x_rotated = x * c - y * s
    y_rotated = x * s + y * c

    return x_rotated, y_rotated


def sign(number: float) -> Literal[-1, 0, 1]:
    """Return 1 for positive number, -1 for negative number, 0 for 0."""
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0


def euclidean_distance(
    p1: Tuple[float, float, float], p2: Tuple[float, float, float]
) -> float:
    """Helper method that returns the Euclidean distance between points p1 and p2, each
    in form (x, y, yaw).
    """
    return ((p1[0] - p2[0]) ** 2.0 + (p1[1] - p2[1]) ** 2.0) ** 0.5


def wrap_to_pi(angle: float) -> float:
    """Wraps the given angle to its equivalent angle within the range -pi to pi."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi

    return angle


########################################################################################
# CURVE HELPER FUNCTIONS ###############################################################
########################################################################################


def steering_angles(phi: float, turn_rad: float) -> Tuple[float, float]:
    """Given turning angle (phi) and turn radius, output the steering params for curve
    equations.
    """
    cos_phi_minus = turn_rad * (np.cos(phi) - 1)
    cos_phi_plus = turn_rad * (np.cos(phi) + 1)

    return cos_phi_minus, cos_phi_plus


def polar(x: float, y: float) -> Tuple[float, float]:
    """Return the polar coordinates (r, theta) of the given point (x, y)."""
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    return r, theta
