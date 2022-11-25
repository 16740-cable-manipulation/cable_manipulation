from cable_manipulation import CableManipulation
import sys
import cv2
import numpy as np
import math


class Discretize:
    def __init__(self, cableMask):
        self.windowSize = 55  # not in use
        self.cableMask = cableMask
        self.maskH = np.shape(self.cableMask)[0]
        self.maskW = np.shape(self.cableMask)[1]
        idx = np.argwhere(self.cableMask > 0)
        self.startPosition = idx[0]
        self.resultPixels = []
        self.rim_offset = (
            0  # for cropping the center area before selecting grasp point
        )
        self.vector_grid_size = 65  # for computing the vector

        dh = self.rim_offset
        dw = self.rim_offset
        self.mask_rimOff = self.cableMask[
            dh : self.maskH - dh, dw : self.maskW - dw
        ]
        self.idx = np.argwhere(self.mask_rimOff > 0)

        self.discretized_r = None
        self.discretized_c = None
        self.prev_vec = np.array([0.0, -1.0])  # in np coord, points downward

    def generate_edge_coords(self, w, h):
        edgeA = np.vstack((np.zeros(w), np.arange(w))).T
        edgeB = np.vstack((np.arange(1, h), np.ones(h - 1) * (w - 1))).T
        edgeC = np.vstack(
            (np.ones(w - 1) * (h - 1), np.arange(w - 2, -1, -1))
        ).T
        edgeD = np.vstack((np.arange(h - 2, 0, -1), np.zeros(h - 2))).T
        edges = np.vstack((edgeA, edgeB, edgeC, edgeD))
        return edges

    def findMeanPixel(self, neighborhood, contours):
        n_shape = neighborhood.shape
        if len(n_shape) == 3:
            neighborhood = neighborhood.mean(-1)

        if len(contours) == 1:
            print("1 island")
            mesh_x, mesh_y = np.meshgrid(
                list(range(len(neighborhood[0]))),
                list(range(len(neighborhood))),
            )
            mean_x = int(np.sum(mesh_x * neighborhood) / np.sum(neighborhood))
            mean_y = int(np.sum(mesh_y * neighborhood) / np.sum(neighborhood))
            # Need to add X,Y offsets after values are returned
            return [mean_x, mean_y]

        elif len(contours) == 2:
            print("2 islands")
            res = []
            ws = []
            hs = []
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                temp_neighborhood = neighborhood[y : y + h, x : x + w]
                mesh_x, mesh_y = np.meshgrid(list(range(w)), list(range(h)))
                mean_x = int(
                    np.sum(mesh_x * temp_neighborhood)
                    / np.sum(temp_neighborhood)
                )
                mean_y = int(
                    np.sum(mesh_y * temp_neighborhood)
                    / np.sum(temp_neighborhood)
                )
                res.append((x + mean_x, y + mean_y))
                ws.append(w)
                hs.append(h)
            weighted_x = int(
                (float(res[0][0]) * ws[0] + float(res[1][0]) * ws[1])
                / (ws[0] + ws[1])
            )
            weighted_y = int(
                (float(res[0][1]) * hs[0] + float(res[1][1]) * hs[1])
                / (hs[0] + hs[1])
            )

            return [weighted_x, weighted_y]

        raise NotImplementedError("Case with contours >2 is not implemented")

    def findVector(self, neighborhood, idx_neighborhood):
        p1 = None
        p1_idx = 0
        p2 = None
        if (
            len(idx_neighborhood) < 2
            or neighborhood.shape[0] <= 2
            or neighborhood.shape[1] <= 2
        ):
            return None
        neighborhood_w = neighborhood.shape[1]
        neighborhood_h = neighborhood.shape[0]
        edges = self.generate_edge_coords(neighborhood_w, neighborhood_h)
        for idx in range(len(edges)):
            tmp_r = int(edges[idx, 0])
            tmp_c = int(edges[idx, 1])
            if neighborhood[tmp_r, tmp_c] == 255:
                if p1 is None:
                    p1 = np.array([tmp_r, tmp_c])
                    p1_idx = idx
                else:
                    if idx - p1_idx > max(neighborhood_w, neighborhood_h):
                        p2 = np.array([tmp_r, tmp_c])
                        break
        if p1 is None or p2 is None:
            return None
        vec = np.floor((p2 - p1)).astype(np.int32)  # in np coord
        print(p1, p2)
        return vec

    def visualize_window(self, mask_grabOK, window_bound, pt=None, vec=None):
        """Visualize a window on the mask

        ``window_bound``: bounding rectangle of the window ((pt1),(pt2))
        ``pt``: the mean pixel of the window, in mask coordinate
        ``vec``: vector starting at ``pt``, tangent to the cable
        """
        mask_tmp = cv2.cvtColor(
            mask_grabOK.astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        mask_tmp = cv2.rectangle(
            mask_tmp, window_bound[0], window_bound[1], (0, 0, 255), 4
        )
        if pt is not None and vec is not None:
            ptprime = [int(pt[0] + vec[0]), int(pt[1] + vec[1])]
            mask_tmp = cv2.line(mask_tmp, pt, ptprime, (0, 255, 0), 4)
            mask_tmp = cv2.circle(mask_tmp, pt, 4, (255, 0, 255), -1)

        cv2.imshow("image", mask_tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findContours(self, neighborhood):
        contours, hierarchy = cv2.findContours(
            neighborhood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return contours

    def isEndCable(self, neighborhood):
        # Assumes binary mask
        return (
            neighborhood.shape[0] == 1
            or neighborhood.shape[1] == 1
            or np.sum(neighborhood) == 0
        )

    def computeInitRC(self, r, c):
        gridSize = self.vector_grid_size
        init_r = min(max(0, r - int((gridSize - 1) / 2)), self.maskH - 1)
        init_c = min(max(0, c - int((gridSize - 1) / 2)), self.maskW - 1)

        return init_r, init_c

    def computeEndRC(self, init_r, init_c):
        gridSize = self.vector_grid_size

        end_r = min(max(0, init_r + gridSize), self.maskH)
        end_c = min(max(0, init_c + gridSize), self.maskW)

        return end_r, end_c

    def unit_vector(self, vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        angle_between((1, 0, 0), (1, 0, 0))
        0.0
        angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def slideWindow(self):
        gridSize = self.vector_grid_size
        # TODO check where the crossings are
        while True:
            if self.discretized_r == None:
                id = np.random.choice(np.arange(self.idx.shape[0]))
                r = self.idx[id, 0]  # row coordinate in the cropped mask
                c = self.idx[id, 1]
                init_r, init_c = self.computeInitRC(r, c)
                end_r, end_c = self.computeEndRC(init_r, init_c)

            else:
                init_r, init_c = self.computeInitRC(
                    self.discretized_r, self.discretized_c
                )
                end_r, end_c = self.computeEndRC(init_r, init_c)
            # TODO should crop the cableMask, not mask_rimOff
            neighborhood = self.mask_rimOff[init_r:end_r, init_c:end_c]
            if self.isEndCable(neighborhood):
                break

            # later on filter out small countours
            contours = self.findContours(neighborhood)
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

            if len(contours) == 1:
                # TODO issue: if the cable is almost horizontal/verticle, the
                # island will be very thin
                island = neighborhood[y : y + h, x : x + w]
                idx_island = np.argwhere(island > 0)
                vec1 = self.findVector(island, idx_island)
            else:
                idx_neighborhood = np.argwhere(neighborhood > 0)
                vec1 = self.findVector(neighborhood, idx_neighborhood)
            if vec1 is None:
                break
            vec1 = vec1.astype(np.float64)
            vec2 = -vec1
            angle1 = self.angle_between(vec1, self.prev_vec)
            angle2 = self.angle_between(vec2, self.prev_vec)
            if angle1 < angle2:
                vec = vec1
            else:
                vec = vec2
            print(angle1, angle2, vec1, vec2)

            self.prev_vec = vec  # here vec is float
            centerR = int((init_r + end_r - 1) / 2)  # int((pt[0] + vec[0])/2 )
            centerC = int((init_c + end_c - 1) / 2)  # int((pt[1] + vec[1])/2)
            unitV = self.unit_vector(vec)
            self.discretized_r = centerR + int(unitV[0] * gridSize)
            self.discretized_c = centerC + int(unitV[1] * gridSize)

            # Need to add offset to the mean pixel coordinates
            mean_pixel = self.findMeanPixel(
                neighborhood, contours
            )  # x is col, y is row
            mean_pixel[0] += init_c
            mean_pixel[1] += init_r
            self.resultPixels.append(mean_pixel)
            self.visualize_window(
                self.cableMask,
                [[init_c, init_r], [end_c, end_r]],
                mean_pixel,
                [int(unitV[1] * gridSize), int(unitV[0] * gridSize)],
            )
        self.visualize_window(
            self.cableMask, [[init_c, init_r], [end_c, end_r]]
        )
        print("exited from slide window")


def getCablesDataFromImage(self, img):
    """Generate cable data dictionary given an cv BGR image"""
    cables_data = {}

    cable_manipulator = CableManipulation(640, 480, use_rs=False)
    available_masks = cable_manipulator.get_available_masks(img)
    for color, mask in available_masks.items():
        disc = Discretize(mask)
        disc.slideWindow()
        data = {
            "coords": disc.resultPixels,
            # TODO
            "pos": disc.pos,
            "cx": disc.cx,
            "color": color,
        }
        cables_data[color] = data
    return cables_data


if __name__ == "__main__":
    img = cv2.imread("cableImages/rs_cable_imgs/img007.png")
    cable_manipulator = CableManipulation(640, 480, use_rs=False)
    available_masks = cable_manipulator.get_available_masks(img)
    yellow_mask = available_masks["blue"]
    disc = Discretize(yellow_mask)
    disc.slideWindow()

