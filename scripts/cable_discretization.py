from cable_manipulation import CableManipulation
import sys
import cv2
import numpy as np
import math


class Discretize:
    def __init__(self, cableMask):
        self.windowSize = 45
        self.cableMask = cableMask
        self.maskH = np.shape(self.cableMask)[0]
        self.maskW = np.shape(self.cableMask)[1]
        idx = np.argwhere(self.cableMask > 0)
        self.startPosition = idx[0]
        self.resultPixels = []
        self.rim_offset = (
            100  # for cropping the center area before selecting grasp point
        )
        self.vector_grid_size = 41  # for computing the vector

        dh = self.rim_offset
        dw = self.rim_offset
        self.mask_rimOff = self.cableMask[
            dh : self.maskH - dh, dw : self.maskW - dw
        ]
        self.idx = np.argwhere(self.mask_rimOff > 0)

        edgeA = np.vstack(
            (np.zeros(self.vector_grid_size), np.arange(self.vector_grid_size))
        ).T
        edgeB = np.vstack(
            (
                np.arange(1, self.vector_grid_size),
                np.ones(self.vector_grid_size - 1)
                * (self.vector_grid_size - 1),
            )
        ).T
        edgeC = np.vstack(
            (
                np.ones(self.vector_grid_size - 1)
                * (self.vector_grid_size - 1),
                np.arange(self.vector_grid_size - 2, -1, -1),
            )
        ).T
        edgeD = np.vstack(
            (
                np.arange(self.vector_grid_size - 2, 0, -1),
                np.zeros(self.vector_grid_size - 2),
            )
        ).T
        self.edges = np.vstack((edgeA, edgeB, edgeC, edgeD))

        self.discretized_r = None
        self.discretized_c = None

    def findMeanPixel(self, neighborhood, contours):
        n_shape = neighborhood.shape
        if len(n_shape) == 3:
            neighborhood = neighborhood.mean(-1)

        if len(contours) == 1:
            mesh_x, mesh_y = np.meshgrid(
                list(range(len(neighborhood[0]))),
                list(range(len(neighborhood))),
            )
            mean_x = int(np.sum(mesh_x * neighborhood) / np.sum(neighborhood))
            mean_y = int(np.sum(mesh_y * neighborhood) / np.sum(neighborhood))
            # Need to add X,Y offsets after values are returned
            return [mean_x, mean_y]

        elif len(contours) == 2:
            res = []
            box1, box2 = contours  # (x,y,w,h)
            for b in [box1, box2]:
                x, y, w, h = b
                temp_neighborhood = neighborhood[y : y + h, x : x + w]
                mesh_x, mesh_y = np.meshgrid(list(range(h)), list(range(w)))
                mean_x = int(
                    np.sum(mesh_x * temp_neighborhood)
                    / np.sum(temp_neighborhood)
                )
                mean_y = int(
                    np.sum(mesh_y * temp_neighborhood)
                    / np.sum(temp_neighborhood)
                )
                res.append((x + mean_x, y + mean_y))

            return [(res[0][0] + res[1][0]) // 2, (res[0][1] + res[1][1]) // 2]

        raise NotImplementedError("Case with contours >2 is not implemented")

    def findVector(self, neighborhood, idx_neighborhood):

        # if there is more than 1 pixel in this neighbohood and the distance between these neighboods are over a threshold value
        # out the pixel pair in a list
        gridSize = self.vector_grid_size
        p1 = None
        p1_idx = 0
        p2 = None
        if len(idx_neighborhood) < 2:
            return None
        print(len(idx_neighborhood))

        for idx in range(len(self.edges)):
            tmp_r = int(self.edges[idx, 0])
            tmp_c = int(self.edges[idx, 1])
            if neighborhood[tmp_r, tmp_c] > 0:
                if p1 is None:
                    p1 = np.array([tmp_r, tmp_c])
                    p1_idx = idx
                else:
                    if idx - p1_idx > gridSize:
                        p2 = np.array([tmp_r, tmp_c])
        if p1 is None or p2 is None:
            return None
        vec = np.floor(np.flip((p2 - p1)))

        ptprime = pt + vec
        if (
            ptprime[0] < 0
            or ptprime[0] >= self.maskW
            or ptprime[1] < 0
            or ptprime[1] >= self.maskH
        ):
            vec = -vec
        pt = (int(pt[0]), int(pt[1]))
        vec = (int(vec[0]), int(vec[1]))

        return pt, vec

    def visualize_vector(self, mask_grabOK, pt, ptprime):
        mask_tmp = cv2.cvtColor(
            mask_grabOK.astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        mask_with_vec = cv2.line(mask_tmp, pt, ptprime, (0, 255, 0), 4)
        # plt.imshow(mask_with_vec)
        cv2.imshow("image", mask_with_vec)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findContours(self, neighborhood):
        contours, hierarchy = cv2.findContours(
            neighborhood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return contours

    def isEndCable(self, neighborhood):
        # Assumes binary mask

        return np.sum(neighborhood) == 0

    def computeInitRC(self, r, c):
        gridSize = self.vector_grid_size
        init_r = min(
            max(0, r - int((gridSize - 1) / 2)),
            self.maskH - 1,
        )
        init_c = min(
            max(0, c - int((gridSize - 1) / 2)),
            self.maskW - 1,
        )

        return init_r, init_c

    def computeEndRC(self, init_r, init_c):
        gridSize = self.vector_grid_size

        end_r = min(
            max(0, init_r + gridSize),
            self.maskH,
        )
        end_c = min(
            max(0, init_c + gridSize),
            self.maskW,
        )

        return end_r, end_c

    def unit_vector(self, vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
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

            neighborhood = self.mask_rimOff[init_r:end_r, init_c:end_c]
            idx_neighborhood = np.argwhere(neighborhood > 0)

            if self.isEndCable(neighborhood):
                break

            # later on filter out small countours
            contours = self.findContours(neighborhood)
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

            if len(contours) == 1:
                island = neighborhood[y : y + h, x : x + w]
                idx_island = np.argwhere(island > 0)
                pt, vec1 = self.findVector(island, idx_island)
            else:
                pt, vec1 = self.findVector(neighborhood, idx_neighborhood)

            vec2 = -vec1
            angle1 = self.angle_between(vec1, self.prev_vec)
            angle2 = self.angle_between(vec2, self.prev_vec)
            if angle1 < angle2:
                vec = vec1
            else:
                vec = vec2

            self.prev_vec = vec
            centerR = int((init_r + end_r - 1) / 2)  # int((pt[0] + vec[0])/2 )
            centerC = int((init_c + end_c - 1) / 2)  # int((pt[1] + vec[1])/2)
            unitV = self.unit_vector(vec)
            self.discretized_r = centerR + int(unitV[0] * gridSize)
            self.discretized_c = centerC + int(unitV[1] * gridSize)

            # Need to add offset to the mean pixel coordinates
            mean_pixel = self.findMeanPixel(
                neighborhood, contours
            )  # x is col, y is row

            self.resultPixels.append(mean_pixel)

        return


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
