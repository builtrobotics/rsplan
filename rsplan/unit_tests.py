from typing import List, Tuple

import numpy as np
import pytest

from rsplan import helpers, planner, primitives

_STEP_SIZE = 0.05  # in meters
_RANDOM_PATH_DISTANCE_RANGE = 10  # in meters
_RANDOM_PATH_ANGLE_RANGE = 2 * np.pi  # radians

_ORIGIN = (0.0, 0.0, 0.0)
_TRANSLATED = (4, 7, 0)
_ROTATED = (0, 0, np.pi / 2.0)
_STRAIGHT_TRANSLATION = (10, 0, 0)
_SLIGHTLY_OFFSET = (0, 0.1, 0)
_SLIGHTLY_FORWARD = (-0.1, 0, 0)

_RUNWAY_LENGTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_TURN_RADII = [1, 2, 4, 6]


def _generate_all_paths(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    has_runway: bool = True,
) -> List[primitives.Path]:
    """Generates all Paths for the end pose using the valid turn radii, each runway
    driving direction (forward and backward) and valid runway lengths.
    """
    rs_paths: List[primitives.Path] = []

    if has_runway:
        for turn_radius in _TURN_RADII:
            for runway_length in _RUNWAY_LENGTHS:
                for driving_direction in [1, -1]:
                    rs_nav_path = planner.path(
                        start,
                        end,
                        turn_radius,
                        driving_direction * runway_length,
                        _STEP_SIZE,
                    )
                    rs_paths.append(rs_nav_path)
    else:
        for turn_radius in _TURN_RADII:
            rs_nav_path = planner.path(start, end, turn_radius, 0.0, _STEP_SIZE)
            rs_paths.append(rs_nav_path)

    return rs_paths


def _nav_path(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    *,
    has_runway: bool = True,
) -> primitives.Path:
    """Obtains all possible Reeds-Shepp paths (with different runway lengths, driving
    directions, and turn radii) and choose the one with the shortest total length.
    """
    rs_nav_paths = _generate_all_paths(start, end, has_runway)
    rs_nav_paths.sort(key=lambda x: x.total_length)
    return rs_nav_paths[0]


@pytest.mark.parametrize(
    "start, end",
    [
        (_ORIGIN, _SLIGHTLY_FORWARD),
        (_SLIGHTLY_FORWARD, _ORIGIN),
        (_ORIGIN, _STRAIGHT_TRANSLATION),
        (_STRAIGHT_TRANSLATION, _ORIGIN),
    ],
)
def test_straight_path(
    start: Tuple[float, float, float], end: Tuple[float, float, float]
) -> None:

    nav_path = _nav_path(start, end, has_runway=False)
    assert nav_path is not None

    assert start == nav_path.start_pose
    assert end == nav_path.end_pose

    assert round(nav_path.total_length, 5) == round(
        helpers.euclidean_distance(start, end), 5
    )

    for wp1, wp2 in zip(nav_path.waypoints()[:-1], nav_path.waypoints()[1:]):
        assert wp1.pose_2d_tuple != wp2.pose_2d_tuple

    assert len(set(wp.driving_direction for wp in nav_path.waypoints())) == 1


@pytest.mark.parametrize("yaw_angle", [0, np.pi / 10.0, np.pi, np.pi * 11.0 / 10.0])
def test_backward_path(yaw_angle: float) -> None:
    end = (-4, -5, yaw_angle)
    nav_path = _nav_path(_ORIGIN, end)
    assert nav_path is not None
    assert nav_path.total_length >= helpers.euclidean_distance(_ORIGIN, end)
    for wp1, wp2 in zip(nav_path.waypoints()[:-1], nav_path.waypoints()[1:]):
        assert wp1.pose_2d_tuple != wp2.pose_2d_tuple

    nav_path2 = _nav_path(_ROTATED, end)
    assert nav_path2 is not None
    assert nav_path.total_length >= helpers.euclidean_distance(_ORIGIN, end)
    for wp1, wp2 in zip(nav_path2.waypoints()[:-1], nav_path2.waypoints()[1:]):
        assert wp1.pose_2d_tuple != wp2.pose_2d_tuple


@pytest.mark.parametrize(
    "start, end",
    [
        (_ORIGIN, _TRANSLATED),
        (_TRANSLATED, _ORIGIN),
        (_ORIGIN, _SLIGHTLY_OFFSET),
        (_ORIGIN, _ROTATED),
    ],
)
def test_path_no_runway(
    start: Tuple[float, float, float], end: Tuple[float, float, float]
) -> None:

    nav_path = _nav_path(start, end, has_runway=False)

    assert nav_path.total_length >= helpers.euclidean_distance(start, end)
    assert start == nav_path.start_pose
    assert end == nav_path.end_pose

    assert all(waypoint.is_runway is False for waypoint in nav_path.waypoints())

    for wp1, wp2 in zip(nav_path.waypoints()[:-1], nav_path.waypoints()[1:]):
        assert wp1.pose_2d_tuple != wp2.pose_2d_tuple



@pytest.mark.parametrize("seed", range(100))
def test_random_path(seed: int) -> None:

    if seed <= 10:
        start = _ORIGIN
    else:
        rng0 = np.random.default_rng(seed + 100)
        x0 = _RANDOM_PATH_DISTANCE_RANGE * rng0.uniform(-1, 1)
        y0 = _RANDOM_PATH_DISTANCE_RANGE * rng0.uniform(-1, 1)
        yaw0 = _RANDOM_PATH_ANGLE_RANGE * rng0.random()
        start = (x0, y0, yaw0)

    rng = np.random.default_rng(seed)
    x = _RANDOM_PATH_DISTANCE_RANGE * rng.uniform(-1, 1)
    y = _RANDOM_PATH_DISTANCE_RANGE * rng.uniform(-1, 1)
    yaw = _RANDOM_PATH_ANGLE_RANGE * rng.random()
    end = (x, y, yaw)

    nav_path = _nav_path(start, end)

    assert nav_path.total_length >= helpers.euclidean_distance(start, end)
    assert nav_path.start_pose == start
    assert nav_path.end_pose == end


    for wp1, wp2 in zip(nav_path.waypoints()[:-1], nav_path.waypoints()[1:]):
        wp1_pose = wp1.pose_2d_tuple
        wp2_pose = wp2.pose_2d_tuple
        # No consecutive duplicates
        assert wp1_pose != wp2_pose
        # No individual distance is greater than the step size
        assert round(helpers.euclidean_distance(wp1_pose, wp2_pose), 2) <= _STEP_SIZE


def main() -> None:
    test_random_path()