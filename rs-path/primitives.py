from __future__ import annotations

import dataclasses
import functools
from typing import Any, List, Literal, Optional, Tuple

import numpy as np

import helpers


@dataclasses.dataclass
class Path:
    """Reeds-Shepp path represented as its start/end points, turn radius (in meters),
    and a list of Segments. Additionally contains a step size value (in meters) used to
    calculate the Waypoint representation of the path.
    """

    start_pt: Tuple[float, float, float]
    end_pt: Tuple[float, float, float]
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
        """Convenience function for decomposing the path points into their components
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
        x0, y0, yaw0 = self.start_pt
        path_points: List[Tuple[float, float, float, float, Literal[-1, 1], bool]] = []

        # Calculate list of Waypoint parameter tuples for non-runway segments
        for ix, segment in enumerate(self.segments):
            if self._has_runway and ix == len(self.segments) - 1:  # Runway segment
                seg_points = segment.calc_waypoints(
                    (x0, y0, yaw0), self.step_size, True, end_pt=self.end_pt
                )
                # Remove duplicated runway starting point
                if Waypoint(*path_points[-1]).is_close(Waypoint(*seg_points[0])):
                    seg_points.pop(0)
            else:  # Non-runway segment
                seg_points = segment.calc_waypoints(
                    (x0, y0, yaw0), self.step_size, False
                )

            path_points.extend(seg_points)  # Add segment pts to list of path pts

            # For next segment, set first point to last pt of this segment
            x0, y0, yaw0 = seg_points[-1][0], seg_points[-1][1], seg_points[-1][2]

        # Ensures that the path's last pt equals the provided end pt by appending end pt
        # to the list of path pts if the last pt in the path is not the end pt
        end_pt_to_add = self._end_pt_to_add(path_points[-1])
        path_points.append(end_pt_to_add) if end_pt_to_add is not None else ()

        return [Waypoint(*point) for point in path_points]

    def _end_pt_to_add(
        self, last_path_pt: Tuple[float, float, float, float, Literal[-1, 1], bool]
    ) -> Optional[Tuple[float, float, float, float, Literal[-1, 1], bool]]:
        """Checks if the last path point equals the provided Path end point. It's
        possible for end points to be slightly off the target end pose due to path
        discretization with a non-ideal step size.
        """
        end_pt_with_params = (*self.end_pt, *last_path_pt[3:])
        if not Waypoint(*last_path_pt).is_close(Waypoint(*end_pt_with_params)):
            # Point to append is end point with last 3 parameters from final path point
            return end_pt_with_params

        return None

    def __hash__(self) -> int:
        segment_tuple = tuple(
            (segment.type, segment.direction, segment.length)
            for segment in self.segments
        )
        return hash(
            (
                self.start_pt,
                self.end_pt,
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
        Turns either left (negative), right (positive) or straight (zero).
        """
        return helpers.sign(self.curvature)

    @property
    def pose_2d_tuple(self) -> Tuple[float, float, float]:
        """The X, Y, and yaw of the RSNavPoint as a Tuple."""
        return (self.x, self.y, self.yaw)

    def transform_to(self, end: Waypoint) -> Tuple[float, float, float]:
        """Calculates the X and Y translation and the yaw rotation values needed to
        transform the current point to the end point.
        """
        x_translation = end.x - self.x
        y_translation = end.y - self.y
        yaw_rotation = np.rad2deg(end.yaw - self.yaw)
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
        start_pt: Tuple[float, float, float],
        step_size: float,
        is_runway: bool,
        end_pt: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[Tuple[float, float, float, float, Literal[-1, 1], bool]]:
        """Calculate the parameters needed (x, y, yaw coordinates, and list of segment
        length and curvature) to calculate the list of points used to represent this
        segment.
        """
        if self.is_straight and end_pt != (0.0, 0.0, 0.0):
            # Runway segment with end point passed in for accuracy
            xs, ys, yaws = self._straight_runway_pts(start_pt, end_pt, step_size)
        else:
            # Non-runway segment
            segment_points = self._interpolated(step_size)
            xs, ys, yaws = self._get_segment_coords(start_pt, segment_points)

        return [
            (xs[i], ys[i], yaws[i], self._curvature(), self.direction, is_runway)
            for i in range(len(xs))
        ]

    def _curvature(self) -> float:
        """Radius of curvature for the segment. Based on segment driving direction,
        curve type, and turn radius.
        """
        if self.type == "left":
            return -1.0 / self.turn_radius
        elif self.type == "right":
            return 1.0 / self.turn_radius
        return 0.0

    def _straight_runway_pts(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        step_size: float,
    ) -> Tuple[list[float], list[float], list[float]]:
        """Calculate a straight line of coordinates from the runway start point to the
        runway end point using the yaw angle of the runway end point to ensure the
        runway coordinates are accurate.
        """
        num_coords = int((self.length / step_size) + 2)
        x_coords = (np.linspace(start[0], end[0], num=num_coords, dtype=float)).tolist()
        y_coords = (np.linspace(start[1], end[1], num=num_coords, dtype=float)).tolist()
        yaw_coords = (np.ones(num_coords) * end[2]).tolist()

        return x_coords, y_coords, yaw_coords

    def _get_segment_coords(
        self,
        start: Tuple[float, float, float],
        segment_points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generates the segment's x, y, yaw coordinate lists for each point in the
        interpolated list) using the segment type, turn radius, and start point.
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

        x0, y0, yaw0 = start
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

        seg_pts = np.arange(0, magnitude, step)

        # Add segment endpoint if the list of segment points is not empty
        seg_pts = np.append(seg_pts, [magnitude]) if seg_pts.any() else np.array([0.0])

        return seg_pts
