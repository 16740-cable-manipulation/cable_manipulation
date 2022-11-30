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
