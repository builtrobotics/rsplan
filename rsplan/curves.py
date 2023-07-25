from itertools import chain
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np

from rsplan import helpers, primitives

"""This file outlines curve calculations for the 48 different Reeds-Shepp path types.
A "|" cusp means direction change between the curves (+ to -, - to +).

As a whole, t, u, v are segment parameters generated in each of the curve helper
functions that represent the distance (angular distance for curved segments, linear
distance for straight segments) of their respective segments.

If there are 3 segments (ccc or csc), t is for segment 1, u is for segment 2, and v is
for segment 3.
If there are 4 or 5 segments, the format is:
- t, u, u, v (2nd and 3rd segments have same angular distance) for cccc paths
- t, pi/2, u, v for ccsc paths (2nd segment has angle pi/2)
- t, u, pi/2, v for cscc paths (3rd segment has angle pi/2)
- t, pi/2, u, pi/2, v for ccscc paths (2nd and 4th segments have angle pi/2)

For more detail, see paper page 378 onwards:
https://msp.org/pjm/1990/145-2/pjm-v145-n2-p06-s.pdf


These parameters (t, u, v, or pi/2 depending on the segment) are represented generally
in the Segment class' "distance" parameter.

The t, u, v, and (sometimes) pi/2 parameters, curve type (left, right, straight),
driving direction (-1, 1), and turn radius represent the information required to
generate the segments for a path.


For each of the 12 path formulas, we create 4 paths:
1. Standard: path parameters are left as passed in
2. Reflection: change curve of each segment (right to left, left to right)
3. Time flip: change direction of each segment (+1 to -1, -1 to +1)
4. Reflection + time flip: change direction and curve of each segment

Curve type symbol to corresponding tuple (type, direction):
L+: (left, 1)
L-: (left, -1)
R+: (right, 1)
R-: (right, -1)
S+: (straight, 1)
S-: (straight, -1)

Left means moving along a leftwards (counterclockwise) circle. Right means moving along
a rightwards (clockwise) circle.
"""

r"""Bird's-eye view diagram (X represents the robot facing the +y direction):

    L+                  y                   R+
       <-               ^               ->
          \             |             /
            \           |           /
              \         |         /
                \       |       /
                  \     |     /
                    \   |   /
                     \  |  /
                      \ | /
                       ___
                        ^
     <----------------| X |-----------------> x
                       ___

                      / | \
                     /  |  \
                    /   |   \
                  /     |     \
                /       |       \
              /         |         \
            /           |           \
          /             |             \
       <-               |               ->
    L-                  v                   R-


"""
"""

C S C
0: ["L+", "S+", "L+"] -- csca -> standard
1: ["R+", "S+", "R+"] -- csca -> reflection
2: ["L-", "S-", "L-"] -- csca -> time flip
3: ["R-", "S-", "R-"] -- csca -> reflection + time flip

4: ["L+", "S+", "R+"] -- cscb -> standard
5: ["R+", "S+", "L+"] -- cscb -> reflection
6: ["L-", "S-", "R-"] -- cscb -> time flip
7: ["R-", "S-", "L-"] -- cscb -> reflection + time flip

C | C | C
8 : ["L+", "R-", "L+"] -- c_c_c -> standard
9 : ["R+", "L-", "R+"] -- c_c_c -> reflection
10: ["L-", "R+", "L-"] -- c_c_c -> time flip
11: ["R-", "L+", "R-"] -- c_c_c -> reflection + time flip

C | C C
12: ["L+", "R-", "L-"] -- c_cc -> standard
13: ["R+", "L-", "R-"] -- c_cc -> reflection
14: ["L-", "R+", "L+"] -- c_cc -> time flip
15: ["R-", "L+", "R+"] -- c_cc -> reflection + time flip

C C | C
16: ["L+", "R+", "L-"] -- cc_c -> standard
17: ["R+", "L+", "R-"] -- cc_c -> reflection
18: ["L-", "R-", "L+"] -- cc_c -> time flip
19: ["R-", "L-", "R+"] -- cc_c -> reflection + time flip

C Cu | Cu C
20: ["L+", "R+", "L-", "R-"] -- ccu_cuc -> standard
21: ["R+", "L+", "R-", "L-"] -- ccu_cuc -> reflection
22: ["L-", "R-", "L+", "R+"] -- ccu_cuc -> time flip
23: ["R-", "L-", "R+", "L+"] -- ccu_cuc -> reflection + time flip

C | Cu Cu | C
24: ["L+", "R-", "L-", "R+"] -- c_cucu_c --> standard
25: ["R+", "L-", "R-", "L+"] -- c_cucu_c --> reflection
26: ["L-", "R+", "L+", "R-"] -- c_cucu_c --> time flip
27: ["R-", "L+", "R+", "L-"] -- c_cucu_c --> reflection + time flip

C | C[pi/2] S C
28: ["L+", "R-", "S-", "L-"] --  c_c2sca -> standard
29: ["R+", "L-", "S-", "R-"] --  c_c2sca -> reflection
30: ["L-", "R+", "S+", "L+"] --  c_c2sca -> time flip
31: ["R-", "L+", "S+", "R+"] --  c_c2sca -> reflection + time flip

32: ["L+", "R-", "S-", "R-"] --  c_c2scb -> standard
33: ["R+", "L-", "S-", "L-"] --  c_c2scb -> reflection
34: ["L-", "R+", "S+", "R+"] --  c_c2scb -> time flip
35: ["R-", "L+", "S+", "L+"] --  c_c2scb -> reflection + time flip

C S C2 | C
36: ["L+", "S+", "R+", "L-"] -- csc2_ca -> standard
37: ["R+", "S+", "L+", "R-"] -- csc2_ca -> reflection
38: ["L-", "S-", "R-", "L+"] -- csc2_ca -> time flip
39: ["R-", "S-", "L-", "R+"] -- csc2_ca -> reflection + time flip

40: ["L+", "S+", "L+", "R-"] -- csc2_cb -> standard
41: ["R+", "S+", "R+", "L-"] -- csc2_cb -> reflection
42: ["L-", "S-", "L-", "R+"] -- csc2_cb -> time flip
43: ["R-", "S-", "R-", "L+"] -- csc2_cb -> reflection + time flip

C | C2 S C2 | C
44: ["L+", "R-", "S-", "L-", "R+"] -- c_c2sc2_c -> standard
45: ["R+", "L-", "S-", "R-", "L+"] -- c_c2sc2_c -> reflection
46: ["L-", "R+", "S+", "L+", "R-"] -- c_c2sc2_c -> time flip
47: ["R-", "L+", "S+", "R+", "L-"] -- c_c2sc2_c -> reflection + time flip
"""


_PATHS: List[List[Tuple[Literal["left", "right", "straight"], Literal[-1, 1]]]] = [
    [("left", 1), ("straight", 1), ("left", 1)],
    [("right", 1), ("straight", 1), ("right", 1)],
    [("left", -1), ("straight", -1), ("left", -1)],
    [("right", -1), ("straight", -1), ("right", -1)],
    [("left", 1), ("straight", 1), ("right", 1)],
    [("right", 1), ("straight", 1), ("left", 1)],
    [("left", -1), ("straight", -1), ("right", -1)],
    [("right", -1), ("straight", -1), ("left", -1)],
    [("left", 1), ("right", -1), ("left", 1)],
    [("right", 1), ("left", -1), ("right", 1)],
    [("left", -1), ("right", 1), ("left", -1)],
    [("right", -1), ("left", 1), ("right", -1)],
    [("left", 1), ("right", -1), ("left", -1)],
    [("right", 1), ("left", -1), ("right", -1)],
    [("left", -1), ("right", 1), ("left", 1)],
    [("right", -1), ("left", 1), ("right", 1)],
    [("left", 1), ("right", 1), ("left", -1)],
    [("right", 1), ("left", 1), ("right", -1)],
    [("left", -1), ("right", -1), ("left", 1)],
    [("right", -1), ("left", -1), ("right", 1)],
    [("left", 1), ("right", 1), ("left", -1), ("right", -1)],
    [("right", 1), ("left", 1), ("right", -1), ("left", -1)],
    [("left", -1), ("right", -1), ("left", 1), ("right", 1)],
    [("right", -1), ("left", -1), ("right", 1), ("left", 1)],
    [("left", 1), ("right", -1), ("left", -1), ("right", 1)],
    [("right", 1), ("left", -1), ("right", -1), ("left", 1)],
    [("left", -1), ("right", 1), ("left", 1), ("right", -1)],
    [("right", -1), ("left", 1), ("right", 1), ("left", -1)],
    [("left", 1), ("right", -1), ("straight", -1), ("left", -1)],
    [("right", 1), ("left", -1), ("straight", -1), ("right", -1)],
    [("left", -1), ("right", 1), ("straight", 1), ("left", 1)],
    [("right", -1), ("left", 1), ("straight", 1), ("right", 1)],
    [("left", 1), ("right", -1), ("straight", -1), ("right", -1)],
    [("right", 1), ("left", -1), ("straight", -1), ("left", -1)],
    [("left", -1), ("right", 1), ("straight", 1), ("right", 1)],
    [("right", -1), ("left", 1), ("straight", 1), ("left", 1)],
    [("left", 1), ("straight", 1), ("right", 1), ("left", -1)],
    [("right", 1), ("straight", 1), ("left", 1), ("right", -1)],
    [("left", -1), ("straight", -1), ("right", -1), ("left", 1)],
    [("right", -1), ("straight", -1), ("left", -1), ("right", 1)],
    [("left", 1), ("straight", 1), ("left", 1), ("right", -1)],
    [("right", 1), ("straight", 1), ("right", 1), ("left", -1)],
    [("left", -1), ("straight", -1), ("left", -1), ("right", 1)],
    [("right", -1), ("straight", -1), ("right", -1), ("left", 1)],
    [("left", 1), ("right", -1), ("straight", -1), ("left", -1), ("right", 1)],
    [("right", 1), ("left", -1), ("straight", -1), ("right", -1), ("left", 1)],
    [("left", -1), ("right", 1), ("straight", 1), ("left", 1), ("right", -1)],
    [("right", -1), ("left", 1), ("straight", 1), ("right", 1), ("left", -1)],
]

_PATH_TYPE_INDICES: List[Tuple[int, int, int, int]] = [
    (0, 1, 2, 3),  # csca
    (4, 5, 6, 7),  # cscb
    (8, 9, 10, 11),  # ccc
    (12, 13, 14, 15),  # c_cc
    (16, 17, 18, 19),  # cc_c
    (20, 21, 22, 23),  # ccu_cuc
    (24, 25, 26, 27),  # c_cucu_c
    (28, 29, 30, 31),  # c_c2sca
    (32, 33, 34, 35),  # c_c2scb
    (36, 37, 38, 39),  # csc2_ca
    (40, 41, 42, 43),  # csc2_cb
    (44, 45, 46, 47),  # c_c2sc2_c
]

_NEAR_ZERO_TOL = 1e-12

_PI_DIVS = [(np.pi / 2.0), (np.pi / 2.0), -(np.pi / 2.0), -(np.pi / 2.0)]


def csc(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step_size: float,
    x: float,
    y: float,
    phi: float,
    turn_rad: float,
) -> List[primitives.Path]:
    """Curve-Straight-Curve paths"""
    cos_phi_minus, cos_phi_plus = helpers.steering_angles(phi, turn_rad)
    paths: List[primitives.Path] = []

    csca_params = _gen_path_parameters(
        _csca, _PATH_TYPE_INDICES[0], x, y, phi, cos_phi_minus, turn_rad
    )

    cscb_params = _gen_path_parameters(
        _cscb, _PATH_TYPE_INDICES[1], x, y, phi, cos_phi_plus, turn_rad
    )

    # NOTE: Straight paths will generate some zero length segments. This is due to the
    # way the algorithm generates possible paths.
    all_path_params = _merge_tuples(csca_params, cscb_params)
    for t, u, v, path_ix in zip(*all_path_params):
        path_type = _PATHS[path_ix]
        segment_params = [t, u, v]
        paths.append(
            _create_path(start, end, step_size, segment_params, path_type, turn_rad)
        )

    return paths


def ccc(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step_size: float,
    x: float,
    y: float,
    phi: float,
    turn_rad: float,
) -> List[primitives.Path]:
    """Curve-Curve-Curve paths"""
    cos_phi_minus, _ = helpers.steering_angles(phi, turn_rad)
    paths: List[primitives.Path] = []

    # C | C | C
    ccc_params = _gen_path_parameters(
        _c_c_c, _PATH_TYPE_INDICES[2], x, y, phi, cos_phi_minus, turn_rad
    )

    # C | C C
    c_cc_params = _gen_path_parameters(
        _c_cc, _PATH_TYPE_INDICES[3], x, y, phi, cos_phi_minus, turn_rad
    )

    # C C | C
    cc_c_params = _gen_path_parameters(
        _cc_c, _PATH_TYPE_INDICES[4], x, y, phi, cos_phi_minus, turn_rad
    )

    all_path_params = _merge_tuples(ccc_params, c_cc_params, cc_c_params)
    for t, u, v, path_ix in zip(*all_path_params):
        path_type = _PATHS[path_ix]
        segment_params = [t, u, v]
        paths.append(
            _create_path(start, end, step_size, segment_params, path_type, turn_rad)
        )

    return paths


def cccc(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step_size: float,
    x: float,
    y: float,
    phi: float,
    turn_rad: float,
) -> List[primitives.Path]:
    """Curve-Curve-Curve-Curve paths"""
    _, cos_phi_plus = helpers.steering_angles(phi, turn_rad)
    paths: List[primitives.Path] = []

    # C Cu | Cu C
    ccu_cuc_params = _gen_path_parameters(
        _ccu_cuc, _PATH_TYPE_INDICES[5], x, y, phi, cos_phi_plus, turn_rad
    )

    # C | Cu Cu | C
    c_cucu_c_params = _gen_path_parameters(
        _c_cucu_c, _PATH_TYPE_INDICES[6], x, y, phi, cos_phi_plus, turn_rad
    )

    all_path_params = _merge_tuples(ccu_cuc_params, c_cucu_c_params)
    for t, u, v, path_ix in zip(*all_path_params):
        path_type = _PATHS[path_ix]
        segment_params = [t, u, u, v]
        paths.append(
            _create_path(start, end, step_size, segment_params, path_type, turn_rad)
        )

    return paths


def ccsc(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step_size: float,
    x: float,
    y: float,
    phi: float,
    turn_rad: float,
) -> List[primitives.Path]:
    """Curve-Curve-Straight-Curve paths"""
    cos_phi_minus, cos_phi_plus = helpers.steering_angles(phi, turn_rad)

    paths: List[primitives.Path] = []
    # C | C[pi/2] S C
    c_c2sca_params = _gen_path_parameters(
        _c_c2sca, _PATH_TYPE_INDICES[7], x, y, phi, cos_phi_minus, turn_rad
    )

    c_c2scb_params = _gen_path_parameters(
        _c_c2scb, _PATH_TYPE_INDICES[8], x, y, phi, cos_phi_plus, turn_rad
    )

    all_path_params = _merge_tuples(c_c2sca_params, c_c2scb_params)
    for ix, path_params in enumerate(zip(*all_path_params)):
        t, u, v, path_ix = path_params
        path_type = _PATHS[path_ix]
        segment_params = [t, _PI_DIVS[ix], u, v]
        paths.append(
            _create_path(start, end, step_size, segment_params, path_type, turn_rad)
        )

    return paths


def cscc(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step_size: float,
    x: float,
    y: float,
    phi: float,
    turn_rad: float,
) -> List[primitives.Path]:
    """Curve-Straight-Curve-Curve paths"""
    cos_phi_minus, cos_phi_plus = helpers.steering_angles(phi, turn_rad)

    paths: List[primitives.Path] = []

    # C S C2 | C
    csc2_ca_params = _gen_path_parameters(
        _csc2_ca, _PATH_TYPE_INDICES[9], x, y, phi, cos_phi_minus, turn_rad
    )

    csc2_cb_params = _gen_path_parameters(
        _csc2_cb, _PATH_TYPE_INDICES[10], x, y, phi, cos_phi_plus, turn_rad
    )

    all_path_params = _merge_tuples(csc2_ca_params, csc2_cb_params)
    for ix, path_params in enumerate(zip(*all_path_params)):
        t, u, v, path_ix = path_params
        path_type = _PATHS[path_ix]
        segment_params = [t, u, _PI_DIVS[ix], v]
        paths.append(
            _create_path(start, end, step_size, segment_params, path_type, turn_rad)
        )

    return paths


def ccscc(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step_size: float,
    x: float,
    y: float,
    phi: float,
    turn_rad: float,
) -> List[primitives.Path]:
    """Curve-Curve-Straight-Curve-Curve paths"""
    _, cos_phi_plus = helpers.steering_angles(phi, turn_rad)

    paths: List[primitives.Path] = []

    # C | C2 S C2 | C
    c_c2sc2_c_params = _gen_path_parameters(
        _c_c2sc2_c, _PATH_TYPE_INDICES[11], x, y, phi, cos_phi_plus, turn_rad
    )

    all_path_params = _merge_tuples(c_c2sc2_c_params)
    for ix, path_params in enumerate(zip(*all_path_params)):
        t, u, v, path_ix = path_params
        path_type = _PATHS[path_ix]
        segment_params = [t, _PI_DIVS[ix], u, _PI_DIVS[ix], v]
        paths.append(
            _create_path(start, end, step_size, segment_params, path_type, turn_rad)
        )

    return paths


########################################################################################
# HELPER FUNCTIONS #####################################################################
########################################################################################


def _create_path(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    step_size: float,
    segment_params: List[float],
    path_segment_types: List[
        Tuple[Literal["left", "right", "straight"], Literal[-1, 1]]
    ],
    turn_radius: float,
) -> primitives.Path:
    """Generates the Path given the path characteristics and the necessary information
    (segment parameters, types, and driving direction) to generate the segments in the
    Path.
    """
    path_segments = []
    for segment_param, segment_type in zip(segment_params, path_segment_types):
        path_segments.append(_create_segment(segment_param, segment_type, turn_radius))

    return primitives.Path(
        start_pose=start,
        end_pose=end,
        segments=path_segments,
        turn_radius=turn_radius,
        step_size=step_size,
    )


def _create_segment(
    segment_param: float,
    segment_type: Tuple[Literal["left", "right", "straight"], Literal[-1, 1]],
    turn_radius: float,
) -> primitives.Segment:
    type, dir = segment_type
    length = segment_param if type == "straight" else segment_param * turn_radius

    return primitives.Segment(
        type=type, direction=dir, length=length, turn_radius=turn_radius
    )


def _gen_path_parameters(
    curve_func: Callable,
    path_type_indices: Tuple[int, int, int, int],
    x: float,
    y: float,
    phi: float,
    cos_phi_param: float,
    turn_rad: float,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Generates the list of path parameters needed to calculate path's segments for the
    four distinct Reeds-Shepp path types:
    1. Standard: path parameters are left as passed in
    2. Reflection: change curve of each segment (R to L, L to R)
    3. Time flip: change direction of each segment (+ to -, - to +)
    4. Reflection + time flip: change direction and curve of each segment

    Filters out invalid path parameter sets so that only valid path parameters are
    returned (if a path type is invalid the curve functions return None).
    """
    pos_sin_phi = turn_rad * np.sin(phi)
    neg_sin_phi = -turn_rad * np.sin(phi)

    standard: Tuple[float, ...] = curve_func(
        x, y, phi, pos_sin_phi, cos_phi_param, turn_rad
    )
    reflection: Tuple[float, ...] = curve_func(
        x, -y, -phi, neg_sin_phi, cos_phi_param, turn_rad
    )
    time_flip: Tuple[float, ...] = curve_func(
        -x, y, -phi, neg_sin_phi, cos_phi_param, turn_rad
    )
    reflection_time_flip: Tuple[float, ...] = curve_func(
        -x, -y, phi, pos_sin_phi, cos_phi_param, turn_rad
    )

    # Multiply path parameters by -1 in order to apply the time flip operation
    if time_flip is not None:
        time_flip = tuple(-param for param in time_flip)
    if reflection_time_flip is not None:
        reflection_time_flip = tuple(-param for param in reflection_time_flip)

    curve_params = [standard, reflection, time_flip, reflection_time_flip]
    path_params = list(zip(curve_params, path_type_indices))
    valid_path_params = [params for params in path_params if params[0] is not None]

    if valid_path_params:
        unzipped_curve_params, path_indices = list(zip(*valid_path_params))
        ts, us, vs = list(zip(*unzipped_curve_params))
        return ts, us, vs, list(path_indices)

    return [], [], [], []


def _merge_tuples(
    *tuples: Tuple[List[float], List[float], List[float], List[int]]
) -> Tuple[List[float], List[float], List[float], List[int]]:
    return tuple([*chain.from_iterable(values)] for values in zip(*tuples))  # type: ignore[return-value]


########################################################################################
# CURVE FORMULA FUNCTIONS ##############################################################
########################################################################################

# --------------------------------------------------------------------------------------
# CSC
# --------------------------------------------------------------------------------------


def _csca(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.1: CSC (same turns)
    a = x - rsin
    b = y + rcos
    u, t = helpers.polar(a, b)

    v = helpers.wrap_to_pi(phi - t)
    if t >= 0 and u >= 0 and v >= 0:
        return t, u, v

    return None


def _cscb(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.2: CSC (opposite turns)
    turn_rad_mult2 = 2 * turn_rad

    a = x + rsin
    b = y - rcos
    r, theta = helpers.polar(a, b)

    if r >= turn_rad_mult2:
        u = np.sqrt(r**2 - turn_rad_mult2**2)
        alpha = np.arctan2(turn_rad_mult2, u)
        t = helpers.wrap_to_pi(theta + alpha)
        v = helpers.wrap_to_pi(t - phi)

        if t >= 0 and u >= 0 and v >= 0:
            return t, u, v

    return None


# --------------------------------------------------------------------------------------
# CCC
# --------------------------------------------------------------------------------------


def _c_c_c(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.3: C|C|C
    turn_rad_mult4 = 4 * turn_rad
    a = x - rsin
    b = y + rcos

    if np.abs(a) < _NEAR_ZERO_TOL and np.abs(b) < _NEAR_ZERO_TOL:
        return None

    r, theta = helpers.polar(a, b)
    if r < turn_rad_mult4:
        alpha = np.arccos(r / turn_rad_mult4)
        t = helpers.wrap_to_pi((np.pi / 2.0) + alpha + theta)
        u = helpers.wrap_to_pi(np.pi - 2 * alpha)
        v = helpers.wrap_to_pi(phi - t - u)

        if t >= 0 and u >= 0 and v >= 0:
            return t, u, v

    return None


def _c_cc(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.4 (1): C|CC
    turn_rad_mult4 = 4 * turn_rad
    a = x - rsin
    b = y + rcos

    if np.abs(a) < _NEAR_ZERO_TOL and np.abs(b) < _NEAR_ZERO_TOL:
        return None

    r, theta = helpers.polar(a, b)
    if r <= turn_rad_mult4:
        alpha = np.arccos(r / turn_rad_mult4)
        t = helpers.wrap_to_pi((np.pi / 2.0) + alpha + theta)
        u = helpers.wrap_to_pi(np.pi - 2 * alpha)
        v = helpers.wrap_to_pi(t + u - phi)

        if t >= 0 and u >= 0 and v >= 0:
            return t, u, v

    return None


def _cc_c(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.4 (2): CC|C
    turn_rad_mult2 = 2 * turn_rad
    turn_rad_mult4 = 4 * turn_rad
    a = x - rsin
    b = y + rcos

    if np.abs(a) < _NEAR_ZERO_TOL and np.abs(b) < _NEAR_ZERO_TOL:
        return None

    r, theta = helpers.polar(a, b)
    if r <= turn_rad_mult4:
        u = np.arccos((8 * (turn_rad**2) - r * r) / (8 * (turn_rad**2)))
        sin_u = np.sin(u)

        if np.abs(sin_u) < 0.001:
            sin_u = 0.0

        if np.abs(sin_u) < 0.001 and np.abs(r) < 0.001:
            return None

        alpha = np.arcsin(turn_rad_mult2 * sin_u / r)
        t = helpers.wrap_to_pi((np.pi / 2.0) - alpha + theta)
        v = helpers.wrap_to_pi(t - u - phi)

        if t >= 0 and u >= 0 and v >= 0:
            return t, u, v

    return None


# --------------------------------------------------------------------------------------
# CCCC
# --------------------------------------------------------------------------------------


def _ccu_cuc(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.7: CCu|CuC
    turn_rad_mult2 = 2 * turn_rad
    turn_rad_mult4 = 4 * turn_rad
    a = x + rsin
    b = y - rcos

    if np.abs(a) < _NEAR_ZERO_TOL and np.abs(b) < _NEAR_ZERO_TOL:
        return None

    r, theta = helpers.polar(a, b)
    if r <= turn_rad_mult4:
        if r > turn_rad_mult2:
            alpha = np.arccos((r / 2 - turn_rad) / turn_rad_mult2)
            t = helpers.wrap_to_pi((np.pi / 2.0) + theta - alpha)
            u = helpers.wrap_to_pi(np.pi - alpha)
            v = helpers.wrap_to_pi(phi - t + 2 * u)

        else:
            alpha = np.arccos((r / 2 + turn_rad) / turn_rad_mult2)
            t = helpers.wrap_to_pi((np.pi / 2.0) + theta + alpha)
            u = helpers.wrap_to_pi(alpha)
            v = helpers.wrap_to_pi(phi - t + 2 * u)

        if t >= 0 and u >= 0 and v >= 0:
            return t, u, v

    return None


def _c_cucu_c(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.8: C|CuCu|C
    turn_rad_mult2 = 2 * turn_rad
    a = x + rsin
    b = y - rcos

    if np.abs(a) < _NEAR_ZERO_TOL and np.abs(b) < _NEAR_ZERO_TOL:
        return None

    r, theta = helpers.polar(a, b)
    if r > 6 * turn_rad:
        return None

    va = (5 * (turn_rad**2) - r * r / 4) / (turn_rad_mult2**2)
    if va < 0 or va > 1:
        return None

    u = np.arccos(va)
    sin_u = np.sin(u)
    alpha = np.arcsin(turn_rad_mult2 * sin_u / r)
    t = helpers.wrap_to_pi((np.pi / 2.0) + theta + alpha)
    v = helpers.wrap_to_pi(t - phi)

    if t >= 0 and u >= 0 and v >= 0:
        return t, u, v

    return None


# --------------------------------------------------------------------------------------
# CCSC
# --------------------------------------------------------------------------------------


def _c_c2sca(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.9 (1): C|C[pi/2]SC
    turn_rad_mult2 = 2 * turn_rad
    a = x - rsin
    b = y + rcos
    r, theta = helpers.polar(a, b)
    if r >= turn_rad_mult2:
        u = np.sqrt(r**2 - turn_rad_mult2**2) - turn_rad_mult2
        if u >= 0:
            alpha = np.arctan2(turn_rad_mult2, (u + turn_rad_mult2))
            t = helpers.wrap_to_pi((np.pi / 2.0) + theta + alpha)
            v = helpers.wrap_to_pi(t + (np.pi / 2.0) - phi)

            if t >= 0 and u >= 0 and v >= 0:
                return t, u, v

    return None


def _c_c2scb(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.9 (1): C|C[pi/2]SC
    turn_rad_mult2 = 2 * turn_rad
    a = x + rsin
    b = y - rcos
    r, theta = helpers.polar(a, b)
    if r >= turn_rad_mult2:
        t = helpers.wrap_to_pi((np.pi / 2.0) + theta)
        u = r - turn_rad_mult2
        v = helpers.wrap_to_pi(phi - t - (np.pi / 2.0))

        if t >= 0 and u >= 0 and v >= 0:
            return t, u, v

    return None


# --------------------------------------------------------------------------------------
# CSCC
# --------------------------------------------------------------------------------------


def _csc2_ca(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.9 (2): CSC[pi/2]|C
    turn_rad_mult2 = 2 * turn_rad
    a = x - rsin
    b = y + rcos
    r, theta = helpers.polar(a, b)
    if r >= turn_rad_mult2:
        u = np.sqrt(r**2 - turn_rad_mult2**2) - turn_rad_mult2
        if u >= 0:
            alpha = np.arctan2(u + turn_rad_mult2, turn_rad_mult2)
            t = helpers.wrap_to_pi((np.pi / 2.0) + theta - alpha)
            v = helpers.wrap_to_pi(t - (np.pi / 2.0) - phi)

            if t >= 0 and u >= 0 and v >= 0:
                return t, u, v

    return None


def _csc2_cb(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.10 (2): CSC[pi/2]|C
    turn_rad_mult2 = 2 * turn_rad
    a = x + rsin
    b = y - rcos
    r, theta = helpers.polar(a, b)
    if r >= turn_rad_mult2:
        t = helpers.wrap_to_pi(theta)
        u = r - turn_rad_mult2
        v = helpers.wrap_to_pi(-t - (np.pi / 2.0) + phi)

        if t >= 0 and u >= 0 and v >= 0:
            return t, u, v

    return None


# --------------------------------------------------------------------------------------
# CCSCC
# --------------------------------------------------------------------------------------


def _c_c2sc2_c(
    x: float, y: float, phi: float, rsin: float, rcos: float, turn_rad: float
) -> Optional[Tuple[float, float, float]]:
    # Formula 8.11: C|C[pi/2]SC[pi/2]|C
    turn_rad_mult2 = 2 * turn_rad
    turn_rad_mult4 = 4 * turn_rad
    a = x + rsin
    b = y - rcos
    r, theta = helpers.polar(a, b)
    if r >= turn_rad_mult4:
        u = np.sqrt(r**2 - turn_rad_mult2**2) - turn_rad_mult4
        if u >= 0:
            alpha = np.arctan2(turn_rad_mult2, (u + turn_rad_mult4))
            t = helpers.wrap_to_pi((np.pi / 2.0) + theta + alpha)
            v = helpers.wrap_to_pi(t - phi)

            if t >= 0 and u >= 0 and v >= 0:
                return t, u, v

    return None
