from __future__ import annotations

import dataclasses
import functools
from typing import Any, List, Literal, Tuple

import numpy as np

from rsplan import helpers


@dataclasses.dataclass
class Path:
    """Reeds-Shepp path represented as its start/end poses, turn radius (in meters),
    and a list of Segments. Additionally contains a step size value (in meters) used to
    calculate the Waypoint representation of the path.
    """

    start_pose: Tuple[float, float, float]
    end_pose: Tuple[float, float, float]
    segments: List[Segment]
    turn_radius: float
    step_size: float

    @property
    def start(self) -> Waypoint:
        return self.waypoints()[0]

    @property
    def end(self) -> Waypoint:
        return self.waypoints()[-1]

    @property
    def _has_runway(self) -> bool:
        return self.segments[-1].is_straight

    @functools.cached_property
    def total_length(self) -> float:
        return sum([abs(segment.length) for segment in self.segments])

    @functools.cached_property
    def runway_length(self) -> float:
        if self._has_runway:
            return abs(self.segments[-1].length)
        return 0.0  # Path does not have runway

    @functools.cached_property
    def number_of_cusp_points(self) -> int:
        count = 0
        for p0, p1 in zip(self.waypoints()[:-1], self.waypoints()[1:]):
            if p0.driving_direction != p1.driving_direction:
                count += 1
        return count

    def prune(self, increment: int) -> List[Waypoint]:
        """Returns a pruned list of waypoints that occur at regularly spaced distances."""
        return [
            self.waypoints()[ix] for ix in range(0, len(self.waypoints()), increment)
        ]

    def coordinates_tuple(self) -> Tuple[List[float], List[float], List[float]]:
        """Convenience function for decomposing the path waypoints into their components
        (x, y, yaw).
        """
        x_coords, y_coords, yaw_coords = [], [], []

        for pt in self.waypoints():
            x_coords.append(pt.x)
            y_coords.append(pt.y)
            yaw_coords.append(pt.yaw)

        return x_coords, y_coords, yaw_coords

    @functools.lru_cache
    def waypoints(self) -> List[Waypoint]:
        """Interpolate the path's segments into a list of Waypoints. First compute the
        pure segment points, then stitch to path list of points. For negative segments,
        we find the segment motion in positive discretization, then we adjust the sign
        of the motion in the equations.
        """
        x0, y0, yaw0 = self.start_pose
        path_points: List[Tuple[float, float, float, float, Literal[-1, 1], bool]] = []

        # Calculate list of Waypoint parameter tuples for non-runway segments
        for ix, segment in enumerate(self.segments):
            if self._has_runway and ix == len(self.segments) - 1:  # Runway segment
                seg_points = segment.calc_waypoints(
                    (x0, y0, yaw0), self.step_size, True, end_pose=self.end_pose
                )
            else:  # Non-runway segment
                seg_points = segment.calc_waypoints(
                    (x0, y0, yaw0), self.step_size, False
                )

            # Remove the duplicated start/end waypoint when combining segments
            if ix > 0 and Waypoint(*path_points[-1]).is_close(Waypoint(*seg_points[0])):
                seg_points.pop(0)

            path_points.extend(seg_points)  # Add segment pts to list of path pts

            # For next segment, set first point to last pt in the current path
            x0, y0, yaw0 = path_points[-1][0], path_points[-1][1], path_points[-1][2]

        return [Waypoint(*point) for point in path_points]


    def __hash__(self) -> int:
        segment_tuple = tuple(
            (segment.type, segment.direction, segment.length)
            for segment in self.segments
        )
        return hash(
            (
                self.start_pose,
                self.end_pose,
                self.turn_radius,
                self.step_size,
                segment_tuple,
            )
        )


@dataclasses.dataclass
class Waypoint:
    """A waypoint along a Reeds-Shepp Path, which includes X, Y (position variables) and
    yaw (orientation variable), as well as curvature and driving direction to represent
    characteristics of the segment the point is on in the overall path.
    """

    x: float
    y: float
    yaw: float
    curvature: float
    driving_direction: Literal[-1, 1]
    is_runway: bool

    @property
    def turn_direction(self) -> Literal[-1, 0, 1]:
        """The direction to turn at the Waypoint defined by the right hand rule.
        Turns either left (positive), right (negative) or straight (zero).
        """
        return helpers.sign(self.curvature)

    @property
    def pose_2d_tuple(self) -> Tuple[float, float, float]:
        """The X, Y, and yaw of the Waypoint as a Tuple."""
        return (self.x, self.y, self.yaw)

    def transform_to(self, end: Waypoint) -> Tuple[float, float, float]:
        """Calculates the X and Y translation and the yaw rotation values needed to
        transform the current point to the end point.
        """
        x_translation = end.x - self.x
        y_translation = end.y - self.y
        yaw_rotation = (end.yaw - self.yaw) * 180 / np.pi
        return x_translation, y_translation, yaw_rotation

    def is_close(self, p2: Waypoint) -> bool:
        return [round(a, 5) for a in (self.x, self.y, self.yaw)] == [
            round(b, 5) for b in (p2.x, p2.y, p2.yaw)
        ]

    def __hash__(self) -> int:
        return hash(
            (
                self.x,
                self.y,
                self.yaw,
                self.curvature,
                self.driving_direction,
                self.is_runway,
            )
        )


@dataclasses.dataclass
class Segment:
    """A single segment within a Reeds Shepp path. A segment is described by its type
    (left, right, or straight), and direction. The direction represents whether the car
    is following the curve type forwards or backwards (see diagram in curves.py).
    The length represents the length of the segment (straight segment's length
    is the straight length and curved segment's length is the arc length calculated by
    turn radius * turn angle).
    """

    type: Literal["left", "right", "straight"]
    direction: Literal[-1, 1]
    length: float
    turn_radius: float

    @property
    def is_straight(self) -> bool:
        return self.type == "straight"

    def calc_waypoints(
        self,
        start_pose: Tuple[float, float, float],
        step_size: float,
        is_runway: bool,
        end_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[Tuple[float, float, float, float, Literal[-1, 1], bool]]:
        """Calculate the parameters needed (x, y, yaw coordinates, and list of segment
        length and curvature) to calculate the list of points used to represent this
        segment.
        """
        segment_points = self._interpolated(step_size)
        xs, ys, yaws = self._get_segment_coords(start_pose, segment_points)

        # If it is a runway we want to use the end_pose yaw
        if is_runway:
            yaws = [end_pose[-1] for _ in yaws]

        return [
            (xs[i], ys[i], yaws[i], self._curvature(), self.direction, is_runway)
            for i in range(len(xs))
        ]

    def _curvature(self) -> float:
        """Radius of curvature for the segment. Based on segment driving direction,
        curve type, and turn radius.
        """
        if self.type == "left":
            return 1.0 / self.turn_radius
        elif self.type == "right":
            return -1.0 / self.turn_radius
        return 0.0


    def _get_segment_coords(
        self,
        start_pose: Tuple[float, float, float],
        segment_points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generates the segment's x, y, yaw coordinate lists for each point in the
        interpolated list) using the segment type, turn radius, and start pose.
        """
        if self.type == "left":
            xs = self.direction * self.turn_radius * np.sin(segment_points)
            ys = self.turn_radius * (1 - np.cos(segment_points))
            yaws = self.direction * segment_points
        elif self.type == "right":
            xs = self.direction * self.turn_radius * np.sin(segment_points)
            ys = -self.turn_radius * (1 - np.cos(segment_points))
            yaws = -self.direction * segment_points
        elif self.type == "straight":
            xs = self.direction * segment_points
            ys = np.zeros(xs.shape[0])
            yaws = np.zeros(xs.shape[0])

        x0, y0, yaw0 = start_pose
        # Rotate generic coordinates w.r.t segment start orientation
        yaw_coords = (yaws + yaw0).tolist()
        xs, ys = helpers.rotate(xs, ys, yaw0) if yaw0 != 0 else (xs, ys)

        # Add segment start position (x0 and y0) values to x and y coordinates
        x_coords = (xs + x0).tolist()
        y_coords = (ys + y0).tolist()

        return x_coords, y_coords, yaw_coords

    def _interpolated(
        self, step_size: float
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Discretizes the segment into a list of equidistant points (starting from 0,
        not actual segment starting point).
        """
        magnitude = (
            abs(self.length)
            if self.is_straight
            else abs(self.length) / self.turn_radius
        )
        # step is the distance between points along the segment: dl (linear distance for
        # straight segments) and dtheta (step size / turn radius) for curved segments.
        step = step_size if self.is_straight else step_size / self.turn_radius
        # Calculate num_steps to guarantee at least 2 points (start and end),
        # addressing cases where magnitude < step size.
        num_steps = int((magnitude / step) + 2)
        # Prefer linspace to arange to avoid floating point errors. Will distribute
        # remainder distance across steps instead of having a short final step.
        return np.linspace(0, magnitude, num_steps, endpoint=True)
