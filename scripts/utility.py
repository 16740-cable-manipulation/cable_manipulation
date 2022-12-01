import numpy as np
from shapely.geometry import LineString


def calcDistance(x1, y1, x2, y2):
    result = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return result


def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def is_line_segments_intersect(line1_pt1, line1_pt2, line2_pt1, line2_pt2):
    line1 = LineString(
        [(line1_pt1[0], line1_pt1[1]), (line1_pt2[0], line1_pt2[1])]
    )
    line2 = LineString(
        [(line2_pt1[0], line2_pt1[1]), (line2_pt2[0], line2_pt2[1])]
    )
    return line1.intersects(line2)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    angle_between((1, 0, 0), (1, 0, 0))
    0.0
    angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
