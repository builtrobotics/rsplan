## README

- [Overview](#overview)
- [Usage](#usage)
  - [Initial setup](#installation)
  - [Running](#running)
  - [Demo](#demo)
- [FAQ](#faq)
- [Exhibits](#exhibits)
- [References](#references)

## Overview
This package is a Reeds-Shepp library implementation with Python that is compatible with Python versions including 3.11.

Contains the following files:
  - `planner`: Path planning code. `path` function outputs the optimal Reeds-Shepp path with or without a runway.
  - `curves`: Curve formulas for all of the curve types in the Reeds-Shepp paper.
  - `primitives`: Three class architectures (path, waypoint, and segment).
    - Waypoint class (stores x, y, yaw, as well as the curvature and angle/length of the segment the waypoint is on)
    - Segment class (stores left/right/straight type, forward/backward direction, magnitude (arc angle or straight length), and turn radius of the segment)
    - Path class (stores start and end points, turn radius, step size, and list of Segments. Also contains a cached waypoints function to get a list of Waypoints for the path)
  - `helpers`: Helper functions for planner, primitives, and curves files.
  - `demo`: Demo/visualization of paths.



## Usage

### Running

See demo.py for example usage

path(start_pt, end_pt, turn_radius, runway_length, step_size, length_tolerance (optional))
- return a Reeds-Shepp path from start_pt to end_pt with specified turning radius and step_size between points. The length_tolerance default is 2 meters but can be set to user preference — if paths’ total lengths are within length tolerance of each other, the path function will choose the one with fewer segments as the optimal path. The runway_length is for the runway at the end of the path that helps improve accuracy in reaching the final position — this can be set to 0, or a positive or negative number for a forwards or backwards driving runway.
- start_pt and end_pt are in the format `Tuple[float, float, float]` of x, y, yaw values.

### Demo

$ python demos/demo.py


## FAQ

What are t, u, and v parameters?
- As a whole, t, u, v are segment parameters generated in each of the curve helper functions that represent the distance (angular distance for curved segments, linear distance for straight segments) of their respective segments.
- If there are 3 segments (ccc or csc), t is for segment 1, u is for segment 2, and v is for segment 3.
- If there are 4 or 5 segments, the format is:
  - t, u, u, v for cccc paths (2nd and 3rd segments have same angular distance)
  - t, pi/2, u, v for ccsc paths (2nd segment has angle pi/2)
  - t, u, pi/2, v for cscc paths (3rd segment has angle pi/2)
  - t, pi/2, u, pi/2, v for ccscc paths (2nd and 4th segments have angle pi/2)
- These are represented generally in the Segment class in the "distance" parameter, which we pass in t, u, v, or pi/2 for depending on the segment.


### Exhibits

From demo.py:

Paths start from origin
Format: (end x, end y, yaw, turn radius, runway length)
1. (5, 6, np.pi, 1, 0)
2. (15, 3, np.pi / 2.0, 2, 6)
3. (-2, -4, np.pi, 4, 3)
4. (-7, 2, np.pi, 4, 0)
5. (-7, -7, 0.0, 6, 1)
6. (0.7, 1.8, 1, 1, 1)
7. (-5, 6, np.pi / 3.0, 2, 1)
8. (7, 2, 0.0, 6, 3)
9. (-4, -1, -np.pi / 2.0, 1, 3)

![Screenshot from 2023-07-18 11-50-32](https://github.com/builtrobotics/mariana/assets/44348827/eed5e06c-059e-48cb-9dc3-e56346f84476)


## References
Paper providing more information on the algorithm:
Reeds, J., & Shepp, L. (1990). Optimal paths for a car that goes both forwards and backwards. https://msp.org/pjm/1990/145-2/pjm-v145-n2-p06-s.pdf

Curve formulas can be found on page 390-391.
Inspiration for this Python implementation of the Reeds-Shepp algorithm:
https://github.com/boyali/reeds_and_shepp_curves/tree/master 
