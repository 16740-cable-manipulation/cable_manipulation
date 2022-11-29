from cable_manipulation import CableManipulation
import sys
import cv2
import numpy as np
import math
from graph_builder import POS_DOWN, POS_UP, POS_NONE
import copy


def calcDistance(x1, y1, x2, y2):
    result = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return result


class Discretize:
    def __init__(self, color, available_masks):
        self.cableMask = available_masks[color]
        self.mask_others = {}
        for c, mask in available_masks.items():
            if c != color:
                self.mask_others[c] = mask
        self.maskH = np.shape(self.cableMask)[0]
        self.maskW = np.shape(self.cableMask)[1]
        idx = np.argwhere(self.cableMask > 0)
        self.startPosition = idx[0]
        self.resultPixels = []
        # for cropping the center area before selecting grasp point
        self.rim_offset = 0  # not in use
        self.window_size = 35  # for sliding window

        # dh = self.rim_offset
        # dw = self.rim_offset
        # self.mask_rimOff = self.cableMask[
        #     dh : self.maskH - dh, dw : self.maskW - dw
        # ]
        self.idx = np.argwhere(self.cableMask > 0)

        # where the sliding started. might need to go back and attept the
        # opposite direction
        self.init_slide_r = None
        self.init_slide_c = None
        self.init_slide_vec = None

        self.prev_vec = np.array([1.0, 0.0])  # in np coord, points downward

        self.cx_other_map = {}  # {(x0,y0): {"green"}, (x1,y1): {"green","red"}}
        self.cx = []  # length equals number of crossings
        self.pos = []  # same length as resultPixel

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
            # print("1 island")
            mesh_x, mesh_y = np.meshgrid(
                list(range(len(neighborhood[0]))),
                list(range(len(neighborhood))),
            )
            mean_x = int(np.sum(mesh_x * neighborhood) / np.sum(neighborhood))
            mean_y = int(np.sum(mesh_y * neighborhood) / np.sum(neighborhood))
            # Need to add X,Y offsets after values are returned
            return [mean_x, mean_y]

        elif len(contours) == 2:
            # print("2 islands")
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
                (float(res[0][0]) * ws[1] + float(res[1][0]) * ws[0])
                / (ws[0] + ws[1])
            )
            weighted_y = int(
                (float(res[0][1]) * hs[1] + float(res[1][1]) * hs[0])
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

    def findVector_PCA(self, idx_neighborhood):
        num_component = 0  # first principle axis
        if len(idx_neighborhood) < 2:
            return None
        idx_meaned = idx_neighborhood - np.mean(idx_neighborhood, axis=0)
        cov_mat = np.cov(idx_meaned, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        # in np coord
        vec = sorted_eigenvectors[:, num_component]
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

    def isEndCable(self, neighborhood, idx_neighborhood):
        # Assumes binary mask
        return (
            neighborhood.shape[0] == 1
            or neighborhood.shape[1] == 1
            or len(idx_neighborhood) < 15  # too few white pixels
            or np.sum(neighborhood) == 0
        )

    def computeInitRC(self, r, c):
        gridSize = self.window_size
        init_r = min(max(0, r - int((gridSize - 1) / 2)), self.maskH - 1)
        init_c = min(max(0, c - int((gridSize - 1) / 2)), self.maskW - 1)

        return init_r, init_c

    def computeEndRC(self, init_r, init_c):
        gridSize = self.window_size

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

    def slideWindowOneDir(
        self, _discretized_r=None, _discretized_c=None, _dir=None, vis=False
    ):
        gridSize = self.window_size
        discretized_r = copy.deepcopy(_discretized_r)
        discretized_c = copy.deepcopy(_discretized_c)

        # directly slide if the initial condition is passed in
        if (
            _dir is not None
            and discretized_c is not None
            and discretized_r is not None
        ):
            print("Sliding along the other direction")
            # current window (just for visualization)
            init_r, init_c = self.computeInitRC(discretized_r, discretized_c)
            end_r, end_c = self.computeEndRC(init_r, init_c)
            # visualize current window
            if vis:
                self.visualize_window(
                    self.cableMask,
                    [[init_c, init_r], [end_c, end_r]],
                    [discretized_c, discretized_r],
                    [int(_dir[1] * gridSize), int(_dir[0] * gridSize)],
                )
            # slided window
            discretized_r = self.init_slide_r + int(_dir[0] * gridSize)
            discretized_c = self.init_slide_c + int(_dir[1] * gridSize)

        while True:
            if discretized_r == None:
                id = np.random.choice(np.arange(self.idx.shape[0]))
                r = self.idx[id, 0]  # row coordinate in the cropped mask
                c = self.idx[id, 1]
                init_r, init_c = self.computeInitRC(r, c)
                end_r, end_c = self.computeEndRC(init_r, init_c)

            else:
                init_r, init_c = self.computeInitRC(
                    discretized_r, discretized_c
                )
                end_r, end_c = self.computeEndRC(init_r, init_c)

            neighborhood = self.cableMask[init_r:end_r, init_c:end_c]
            idx_neighborhood = np.argwhere(neighborhood > 0)
            if self.isEndCable(neighborhood, idx_neighborhood):
                break

            # later on filter out small countours
            contours = self.findContours(neighborhood)
            if len(contours) == 1:
                (x, y, w, h) = cv2.boundingRect(contours[0])
                island = neighborhood[y : y + h, x : x + w]
                idx_island = np.argwhere(island > 0)
                vec1 = self.findVector_PCA(idx_island)
            elif len(contours) == 2:
                vec1 = self.findVector_PCA(idx_neighborhood)
            else:
                raise NotImplementedError(
                    "Case with contours >2 is not implemented"
                )
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

            # Need to add offset to the mean pixel coordinates
            mean_pixel = self.findMeanPixel(
                neighborhood, contours
            )  # x is col, y is row
            mean_pixel[0] += init_c
            mean_pixel[1] += init_r
            self.resultPixels.append(mean_pixel)
            # add crossing
            if len(contours) == 2:
                if mean_pixel not in self.cx:
                    self.cx.append(mean_pixel)
                self.pos.append(POS_DOWN)
                # check if there are other cables near the crossing
                self.check_other_masks(mean_pixel, init_r, init_c, end_r, end_c)
            else:
                self.pos.append(POS_NONE)
            # update window
            unitV = self.unit_vector(vec)

            self.prev_vec = copy.deepcopy(unitV)
            if self.init_slide_vec is None:
                self.init_slide_vec = copy.deepcopy(unitV)
                self.init_slide_r = mean_pixel[1]
                self.init_slide_c = mean_pixel[0]

            discretized_r = mean_pixel[1] + int(unitV[0] * gridSize)
            discretized_c = mean_pixel[0] + int(unitV[1] * gridSize)
            if vis:
                self.visualize_window(
                    self.cableMask,
                    [[init_c, init_r], [end_c, end_r]],
                    mean_pixel,
                    [int(unitV[1] * gridSize), int(unitV[0] * gridSize)],
                )
        if vis:
            self.visualize_window(
                self.cableMask, [[init_c, init_r], [end_c, end_r]]
            )
        print("exited from slide window")

    def slideWindow(self, vis=False):
        self.slideWindowOneDir(vis=vis)
        if self.init_slide_vec is not None:
            # reverse previous nodes
            self.resultPixels.reverse()
            self.pos.reverse()
            self.slideWindowOneDir(
                _discretized_r=self.init_slide_r,
                _discretized_c=self.init_slide_c,
                _dir=-self.init_slide_vec,
                vis=vis,
            )
        else:
            raise RuntimeError("cannot slide along the other direction")
   
    def slideWindowTopDown(self,vis = False):
        mid = self.maskW/2
        gridSize = self.window_size
        start_row = int(self.maskH/9)

        origin_mask = np.zeros((self.maskH,self.maskW), dtype="uint8")
        #tmp = int(mid-gridSize/2):int(mid+gridSize/2)
        origin_mask[0:start_row,:] = 255

        start_region = cv2.bitwise_and(self.cableMask,origin_mask)
        if vis:
            cv2.imshow("mask of origin",start_region)
            cv2.waitKey(0) # wait for ay key to exit window
            cv2.destroyAllWindows() # close all windows
        start_idx = np.argwhere(start_region > 0)
        start_id = np.random.choice(np.arange(start_idx.shape[0]))
        fixed_r = self.idx[start_id, 0]  # row coordinate in the cropped mask
        fixed_c = self.idx[start_id, 1]


        self.slideWindowOneDir(
            _discretized_r=fixed_r,
            _discretized_c=fixed_c,
            _dir=None,
            vis = vis)

        self.resultPixels.reverse()
        self.pos.reverse()
        self.slideWindowOneDir(
                _discretized_r=self.init_slide_r,
                _discretized_c=self.init_slide_c,
                _dir=-self.init_slide_vec,
                vis=vis,
            )

        return
    
    def check_other_masks(self, cx, init_r, init_c, end_r, end_c):
        for color, mask in self.mask_others.items():
            neighborhood = mask[init_r:end_r, init_c:end_c]
            if np.sum(neighborhood) > 0:
                cx_tuple = (cx[0], cx[1])
                others_set = self.cx_other_map.get(cx_tuple)
                if others_set is None:
                    self.cx_other_map[cx_tuple] = {color}
                else:
                    self.cx_other_map[cx_tuple].add(color)

    def refine_result(self, cx):
        # find closest pixel to cx (a Tuple) from resultPixels
        best_dist = None
        best_pixel_idx = None
        for i, pixel in enumerate(self.resultPixels):
            dist = calcDistance(pixel[0], pixel[1], cx[0], cx[1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_pixel_idx = i
        # TODO check distance and decide whether to add a new pixel or override
        # now we simply always override
        if True:
            # override
            if self.pos[best_pixel_idx] == POS_NONE:
                self.pos[best_pixel_idx] = POS_UP
                if [cx[0], cx[1]] not in self.cx:
                    self.cx.append([cx[0], cx[1]])
                self.resultPixels[best_pixel_idx] = [cx[0], cx[1]]
        else:
            # add
            pass


def getCablesDataFromImage(img, vis=False):
    """Generate cable data dictionary given an cv BGR image"""
    cables_data = {}
    cables_disc = {}
    img_w = img.shape[1]
    img_h = img.shape[0]
    cable_manipulator = CableManipulation(img_w, img_h, use_rs=False)
    available_masks = cable_manipulator.get_available_masks(img)

    for color in available_masks.keys():
        disc = Discretize(color, available_masks)
        # disc.slideWindow(vis=vis)
        disc.slideWindowTopDown(vis= vis)
        cables_disc[color] = disc
    # refine coordinates
    for color, disc in cables_disc.items():
        for cx, others in disc.cx_other_map.items():
            for other in others:
                cables_disc[other].refine_result(cx)

    for color, disc in cables_disc.items():
        data = {
            "coords": disc.resultPixels,
            "pos": disc.pos,
            "cx": disc.cx,
            "color": color,
            "width": img_w,
            "height": img_h,
        }
        cables_data["cable_" + color] = data
    return cables_data


if __name__ == "__main__":
    # img = cv2.imread("cableImages/rs_cable_imgs/img005.png")
    # img = cv2.imread("d:/XinyuWang/2022_Fall/16740/cable_manipulation/cableImages/rs_cable_imgs/img005.png")
    img = cv2.imread("d:/XinyuWang/2022_Fall/16740/cable_manipulation/cableImages/generated_01.png")


    cable_manipulator = CableManipulation(640, 480, use_rs=False)
    available_masks = cable_manipulator.get_available_masks(img)
    # yellow_mask = available_masks["yellow"]
    # disc = Discretize(yellow_mask)
    disc = Discretize("blue",available_masks)
    disc.slideWindow()
    res = getCablesDataFromImage(img, vis=True)
    print(res)

