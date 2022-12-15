import numpy as np
import random
import cv2
from graph_builder import Graph, CableGraph, reset_id
from graph_builder import POS_DOWN, POS_UP, POS_NONE, NODE_FREE, DPI
from cable_discretization import getCablesDataFromImage
from action import Action

from my_franka import MyFranka
from rs_driver import Realsense
from utility import get_rotation_matrix, unit_vector, calcDistance
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab

UNWEAVE_IN_PROGRESS = 0
UNWEAVE_ALL_DONE = 1
UNWEAVE_FAIL = 2

MODE_ELIM = 0
MODE_REDISTRIB = 1


class CableSimplePolicy:
    def __init__(self, width=640, height=480, use_fa=False, use_rs=False):
        self.cg = CableGraph()
        self.use_fa = use_fa
        if use_fa:
            self.fa = MyFranka()
        self.width = width
        self.height = height
        self.min_edge_dist = 30
        self.workspace = [
            (0, 0),
            (self.width, self.height),
        ]  # top left, bot right
        self.depth = None

        # for MODE_ELIM
        self.weight_dist = 1.0
        self.weight_curv = 100
        self.weight_uncertainty = 200
        # for MODE_REDISTRIB
        self.weight_dist_cov = 50

        self.weight_cost = 1.0
        self.weight_elim = 30

        self.rim = 30  # px
        self.min_lift_z = 0.06
        self.max_lift_z = 0.3
        self.use_rs = use_rs

        self.theta_range_bound = 0.05
        if self.use_rs is True:
            self.realsense = Realsense()

    def gen_graph_from_image(self, img):
        cable_data = getCablesDataFromImage(img, vis=False)
        print(cable_data)
        self.cg.create_graphs(cable_data)
        self.cg.create_compound_graph()

    def eliminate_crossing(self, cableID):
        """Attempt to eliminate the first crossing of a cable.

        Return the action if exists, none o/w
        """
        print("Attempting to eliminate a cx on # ", cableID)
        graph: Graph = self.cg.graphs[cableID]
        self.cg.compound_graph.visualize(
            save_path="cableGraphs/composite_original.png"
        )
        # graph.visualize(save_path=f"cableGraphs/{cableID}.png")
        # print(graph.get_free_endpoint())
        next_id, cx_pos, nodes = graph.get_next_keypoint(
            graph.get_free_endpoint()
        )
        # print(next_id, cx_pos, nodes)
        if cx_pos == POS_DOWN:  # first crossing is undercx
            print("First cx of # ", cableID, " is UX, returning")
            return None
        if graph.is_fixed_endpoint(next_id):  # this cable is already done
            print("First cx is fixed endpoint. ", cableID, " already done")
            return Action(False)

        # test get next keypoint
        next_id, nodes = graph.get_next_fixed_keypoint(
            graph.get_free_endpoint()
        )
        if len(nodes) < 2:
            print("#", cableID, " has too few graspable nodes, returning")
            return None
        nodes = nodes[:-1]  # the last one is the pivot point

        # attemp to move this node to free space
        grasp_point_id, goal_coord, goal_vec = self.search_goal_coord(
            nodes, next_id, cableID, MODE_ELIM
        )
        print("goal: ", grasp_point_id, goal_coord, goal_vec)
        if goal_coord is not None:
            return self.fill_action_params(
                grasp_point_id, next_id, goal_coord, goal_vec, cableID
            )
        print("Could not find an action to eliminate cx on #", cableID)
        return None

    def redistribute_cable(self, cableID):
        """Attempt to straighten and center a cable.

        Return a non-empty action if exists, none o/w
        """
        print("Attempting to redistribute # ", cableID)
        graph: Graph = self.cg.graphs[cableID]
        self.cg.compound_graph.visualize(
            save_path="cableGraphs/composite_original.png"
        )
        next_id, cx_pos, nodes = graph.get_next_keypoint(
            graph.get_free_endpoint()
        )
        # if cx_pos == POS_UP:
        #     # this means we've already checked this cable in elimitate_crossing
        #     print("First cx of # ", cableID, " is OX, returning")
        #     return None
        # elif cx_pos == POS_DOWN or graph.is_fixed_endpoint(next_id):
        #     # first crossing is UX or first keypoint is fixed endpoint
        #     print("First cx of # ", cableID, " is UX or fixed endpoint")
        #     if len(nodes) < 2:
        #         print("#", cableID, " has too few graspable nodes, returning")
        #         return None
        # else:
        #     return None
        print("keypoint:", next_id)
        if len(nodes) < 2:
            print("#", cableID, " has too few graspable nodes, returning")
            return None

        nodes = nodes[:-1]  # the last one is the pivot point
        grasp_point_id, goal_coord, goal_vec = self.search_goal_coord(
            nodes, next_id, cableID, MODE_REDISTRIB
        )
        print("goal: ", grasp_point_id, goal_coord, goal_vec)
        if goal_coord is not None:
            return self.fill_action_params(
                grasp_point_id, next_id, goal_coord, goal_vec, cableID
            )
        print("Could not find an action to redistribute #", cableID)
        return None

    def fill_action_params(
        self, grasp_point_id, pivot_point_id, goal_coord, goal_vec, cableID
    ):
        """Fill in the fields for a non-empty action. If cannot fill in
        some fields due to invalid depth, return None"""
        graph: Graph = self.cg.graphs[cableID]
        # fill in action params (2d, except z)
        action = Action(True)
        pick_point = graph.get_node_coords(grasp_point_id)
        action.pick_coord = pick_point
        action.place_coord = goal_coord
        pivot_point = graph.get_node_coords(pivot_point_id)
        print("pivot_point: ", pivot_point)
        print("place point: ", pick_point, "goal point: ", goal_coord)
        px_dist_pivot_pick = calcDistance(
            pick_point[0], pick_point[1], pivot_point[0], pivot_point[1]
        )
        px_dist_pivot_place = calcDistance(
            goal_coord[0], goal_coord[1], pivot_point[0], pivot_point[1]
        )
        print(
            "px_dist_pivot_place: ",
            px_dist_pivot_place,
            "px_dist_pivot_pick: ",
            px_dist_pivot_pick,
        )
        tmp_px_dist = np.sqrt(
            np.abs(px_dist_pivot_place**2 - px_dist_pivot_pick**2)
        )
        if self.use_rs:
            action.z = np.clip(
                self.px_length_to_m(cableID, tmp_px_dist) * 0.85,  # TODO magic
                self.min_lift_z,
                self.max_lift_z,
            )
        else:
            action.z = self.min_lift_z

        # the direction vector is tangent to cable at grasp point
        action.pick_vec = graph.calc_tangent_vec(grasp_point_id)
        action.place_vec = goal_vec

        if self.use_rs is False:
            return action
        # fill in 3d params
        pick_point_3d_c = self.realsense.deproject_pixel(
            self.depth, action.pick_coord[0], action.pick_coord[1]
        )
        place_point_3d_c = self.realsense.deproject_pixel(
            self.depth, action.place_coord[0], action.place_coord[1]
        )
        if (
            self.is_bad_3d_coord(pick_point_3d_c) is True
            or self.is_bad_3d_coord(place_point_3d_c) is True
        ):
            return None
        action.pick_3d = pick_point_3d_c
        action.place_3d = place_point_3d_c
        # for the direction vector, directly use the 2d vector
        action.pick_vec_3d = np.array(
            [action.pick_vec[0], action.pick_vec[1], 0]
        )
        action.place_vec_3d = np.array(
            [action.place_vec[0], action.place_vec[1], 0]
        )
        return action

    def get_cable_segment_length(self, cableID, pivot_point, goal_coord):
        """Compute cable length in 3D. Require use_rs to be true"""
        assert self.use_rs is True
        len_px = calcDistance(
            pivot_point[0], pivot_point[1], goal_coord[0], goal_coord[1]
        )
        return self.px_length_to_m(cableID, len_px)

    def px_length_to_m(self, cableID, len_px):
        """Deproject cable to 3D. Require use_rs to be true"""
        # get a random edge
        assert self.use_rs is True
        graph: Graph = self.cg.graphs[cableID]
        edges = graph.get_edges()
        edge_idx = np.random.choice(np.arange(len(edges)))
        edge = edges[edge_idx]
        edge_px1 = graph.get_node_coords(edge[0])
        edge_px2 = graph.get_node_coords(edge[1])
        edge_len_px = calcDistance(
            edge_px1[0], edge_px1[1], edge_px2[0], edge_px2[1]
        )
        edge_pt1 = self.realsense.deproject_pixel(
            self.depth, edge_px1[0], edge_px1[1]
        )
        edge_pt2 = self.realsense.deproject_pixel(
            self.depth, edge_px2[0], edge_px2[1]
        )
        if (
            self.is_bad_3d_coord(edge_pt1) is True
            or self.is_bad_3d_coord(edge_pt2) is True
        ):
            return -1
        edge_len_m = calcDistance(
            edge_pt1[0], edge_pt1[1], edge_pt2[0], edge_pt2[1]
        )
        length = edge_len_m / edge_len_px * len_px
        return length

    def get_zero_theta_vector_angle(
        self, grasp_point: np.ndarray, pivot_point: np.ndarray
    ):
        zero_vec = unit_vector(grasp_point - pivot_point)
        return np.arctan2(zero_vec[1], zero_vec[0])

    def get_result_point(
        self, theta, length, grasp_point: np.ndarray, pivot_point: np.ndarray
    ) -> np.ndarray:
        """Result point is generated by rotating a straight line with ``length``
        by ``theta`` radians around the ``pivot_point``, starting from a
        zero theta direction determined by ``grasp_point`` and ``pivot_point``
        """
        zero_vec = unit_vector(grasp_point - pivot_point)
        # rotate zero_vec about the pivot point by theta
        rotated_vec = np.matmul(
            get_rotation_matrix(theta), zero_vec.reshape((-1, 1)).flatten()
        )
        res_point = pivot_point + rotated_vec * length
        return res_point  # not rounded, dtype is float

    def is_in_workspace(self, point):
        return (
            point[0] >= self.workspace[0][0] + self.rim
            and point[0] < self.workspace[1][0] - self.rim
            and point[1] >= self.workspace[0][1] + self.rim
            and point[1] < self.workspace[1][1] - self.rim
        )

    def generate_action_space(
        self, grasp_point_id, pivot_point_id, grasp_length, cableID, vis=True
    ):
        """Return a dict {elim: [[th_start1, th_end1], [th_start2, th_end2]],..}
        and the max_elim_num.
        The actions in the action space doesn't have to eliminate a cx.
        However,
        1. it shouldn't introduce more crossings.
        2. it shouldn't exceed the workspace.
        3. it shouldn't be to close to any keypoints.
        """
        # assume we limit the theta to -pi/2 ~ pi/2 rad
        # note: positive theta is clockwise (cuz it's in cv coord)
        # theta=0 is the line connecting pivot point to grasp point

        graph: Graph = self.cg.graphs[cableID]
        grasp_point = np.array(graph.get_node_coords(grasp_point_id))
        pivot_point = np.array(graph.get_node_coords(pivot_point_id))
        thetas = np.arange(-np.pi / 2, np.pi / 2, 0.03)
        thetas = np.hstack(([0], thetas))
        theta_ranges = {}  # {elim_num1: [[],[],..], elim_num2: [[],[],..]}
        theta_range_tmp = []
        elim_num = 0

        for i, theta in enumerate(thetas):
            res_ok, tmp_elim_num = self.is_action_valid(
                theta, grasp_point_id, pivot_point_id, cableID
            )
            # whenever resok is turned False (or reaches upper theta bound) or
            # elim_num has changed, if we have a theta subrange with sufficient
            # size, add it to the theta ranges dict
            # print(
            #     f"theta: {theta}, res_ok: {res_ok}, tmp_elim_num: {tmp_elim_num}"
            # )
            if (
                res_ok is False
                or i == thetas.shape[0] - 1
                or tmp_elim_num != elim_num
            ):
                if len(theta_range_tmp) == 2:
                    # shrink range before appending
                    theta_range_tmp = self.bound_theta_range(theta_range_tmp)
                    if theta_range_tmp is not None:
                        if theta_ranges.get(elim_num) is None:
                            theta_ranges[elim_num] = []
                        theta_ranges[elim_num].append(theta_range_tmp)
                theta_range_tmp = []
                elim_num = 0
            if res_ok is True:
                elim_num = tmp_elim_num
                if len(theta_range_tmp) < 2:
                    theta_range_tmp.append(theta)
                else:
                    theta_range_tmp[1] = theta

            if i == 0:
                res_ok = False
                theta_range_tmp = []
                elim_num = 0

        if not theta_ranges:
            return None, None
        # only return the theta ranges with the biggest elim num
        max_elim_num = np.max(list(theta_ranges.keys()))
        # draw on fig the theta ranges
        angle0 = self.get_zero_theta_vector_angle(grasp_point, pivot_point)
        # for plotting

        if vis is True:
            self.plot_action_space(
                theta_ranges,
                pivot_point,
                grasp_length,
                angle0,
                pivot_point_id,
                grasp_point_id,
                cableID,
            )
        return theta_ranges, max_elim_num
        # return theta_ranges[max_elim_num]

    def is_action_valid(self, theta, grasp_point_id, pivot_point_id, cableID):
        """Check whether an action is valid.
        1. it shouldn't introduce more crossings.
        2. the res grasp node shouldn't exceed the workspace.
        3. the new subgraph shouldn't be to close to any keypoints.
        """
        # check whether any part of the graph is outside the workspace
        elim_num = 0
        ws = [
            (self.rim, self.rim),
            (self.width - self.rim, self.height - self.rim),
        ]
        graph_this = self.simulate_next_state(
            theta, grasp_point_id, pivot_point_id, cableID
        )
        # print(graph_this.get_node_coords(grasp_point_id))
        if not graph_this.is_in_ws(grasp_point_id, ws):
            # print("action invalid: res grasp point not in ws")
            return False, elim_num

        if graph_this.is_middle_not_in_ws(ws):
            # print("action invalid: some middle nodes not in ws")
            return False, elim_num
        # the first 2 edges, the last 2 edges, all? cx edges
        dangerous_edges = []
        graph: Graph = self.cg.graphs[cableID]
        graph_free_to_keypoint = graph.build_subgraph(
            graph.get_free_endpoint(), graph.get_succ(pivot_point_id)
        )
        exclude_cx = set(graph_free_to_keypoint.get_crossings())
        # print(exclude_cx)
        for cableid, graph_other in self.cg.graphs.items():
            if cableid != cableID:
                dangerous_edges.extend(graph_other.get_first_n_edges(n=2))
                dangerous_edges.extend(graph_other.get_last_n_edges(n=2))
                dangerous_cx = list(
                    set(graph_other.get_crossings()) - exclude_cx
                )
                dangerous_edges.extend(
                    graph_other.get_crossing_edges(cx=dangerous_cx)
                )

        graph_others: Graph = self.cg.create_compound_graph_except(cableID)
        graph_this_sub = graph_this.build_subgraph(
            graph.get_free_endpoint(), pivot_point_id
        )
        composite_graph = graph_others.compose(graph_this_sub)
        for edge1 in dangerous_edges:
            for edge2 in graph_this_sub.get_edges():
                if (
                    composite_graph.calc_distance_between_edges(edge1, edge2)
                    < self.min_edge_dist
                ):
                    # print("action invalid: subgraph too close to keypoints")
                    # print(edge1, edge2)
                    # composite_graph.visualize()
                    return False, elim_num

        # check #cx
        elim_num = self.calc_cx_elim_num(
            grasp_point_id, pivot_point_id, graph_this, cableID
        )
        if elim_num >= 0:
            return True, elim_num  # elim num could be 0
        else:
            # print("action invalid: introduces more cx")
            return False, elim_num

    def calc_cx_elim_num(
        self, grasp_point_id, pivot_point_id, graph_this: Graph, cableID
    ):
        graph: Graph = self.cg.graphs[cableID]
        graph_others: Graph = self.cg.create_compound_graph_except(cableID)
        graph_orig_sub = graph.build_subgraph(
            graph.get_free_endpoint(), pivot_point_id
        )
        num_cx_orig = graph_orig_sub.get_num_crossings()

        line_seg_pt1 = graph_this.get_node_coords(pivot_point_id)
        line_seg_pt2 = graph_this.get_node_coords(grasp_point_id)
        line_seg_pt3 = graph_this.get_node_coords(graph.get_free_endpoint())
        # two linear segments
        num_cx_new = graph_others.get_num_crossings_with_line_seg(
            line_seg_pt1, line_seg_pt2
        ) + graph_others.get_num_crossings_with_line_seg(
            line_seg_pt2, line_seg_pt3
        )
        elim_num = num_cx_orig - num_cx_new
        return elim_num

    def bound_theta_range(self, theta_range):
        th1 = theta_range[0]
        th2 = theta_range[1]
        th1_bounded = th1 + self.theta_range_bound
        th2_bounded = th2 - self.theta_range_bound
        if th1_bounded >= th2_bounded:
            return None
        else:
            return [th1_bounded, th2_bounded]

    def plot_action_space(
        self,
        theta_ranges,
        pivot_point,
        grasp_length,
        angle0,
        # the rest are for generating the graph
        pivot_point_id,
        grasp_point_id,
        cableID,
    ):
        """Create 3 plots:
        1. the simulated state at theta = 0
        2. the simulated state at a valid theta
        3. the entire valid action space
        """
        graph: Graph = self.cg.graphs[cableID]
        max_elim_num = np.max(list(theta_ranges.keys())) + 2
        min_elim_num = 0
        # cmap = plt.get_cmap("cool", max_elim_num - min_elim_num + 1)

        # plot the simulated state at theta = 0
        graph_this = self.simulate_next_state(
            0.0, grasp_point_id, pivot_point_id, cableID
        )
        graph_others: Graph = self.cg.create_compound_graph_except(cableID)
        # create a better graph for visualization
        graph_this_simple = graph_this.build_subgraph(
            pivot_point_id, graph.get_fixed_endpoint()
        )
        graph_this.copy_node(graph_this_simple, grasp_point_id)
        graph_this.copy_node(graph_this_simple, graph.get_free_endpoint())
        graph_this_simple.add_edge(grasp_point_id, pivot_point_id, POS_NONE)
        graph_this_simple.add_edge(
            graph.get_free_endpoint(), grasp_point_id, POS_NONE
        )
        composite_graph_1 = graph_others.compose(graph_this_simple)
        composite_graph_1.visualize()

        # also plot the simulated state at a valid theta
        rang = theta_ranges[np.max(list(theta_ranges.keys()))][0]
        th = np.random.uniform(rang[0], rang[1])
        print("random angle to plot: ", th)
        graph_this_2 = self.simulate_next_state(
            th, grasp_point_id, pivot_point_id, cableID
        )
        graph_this_2_simple = graph_this_2.build_subgraph(
            pivot_point_id, graph.get_fixed_endpoint()
        )
        graph_this_2.copy_node(graph_this_2_simple, grasp_point_id)
        graph_this_2.copy_node(graph_this_2_simple, graph.get_free_endpoint())
        graph_this_2_simple.add_edge(grasp_point_id, pivot_point_id, POS_NONE)
        graph_this_2_simple.add_edge(
            graph.get_free_endpoint(), grasp_point_id, POS_NONE
        )
        composite_graph_2 = graph_others.compose(graph_this_2_simple)
        composite_graph_2.visualize()

        # plot the entire valid action space
        fig, ax = pylab.subplots(
            1, 1, figsize=(self.width / DPI, self.height / DPI), dpi=DPI
        )
        composite_graph = self.cg.compound_graph
        composite_graph.visualize(ax=ax)

        cmap = pylab.cm.cool  # define the colormap
        # extract all colors from the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N
        )
        # define the bins and normalize
        bounds = np.linspace(
            min_elim_num, max_elim_num, max_elim_num - min_elim_num + 1
        )
        print("theta_ranges: ", theta_ranges)
        print("bounds: ", bounds)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cmap_disc = pylab.get_cmap("cool", len(bounds) - 1)
        # draw arcs
        for elim, theta_range in theta_ranges.items():
            rgb = cmap_disc(elim)
            print(f"elim={elim}, color={rgb}")
            for rang in theta_range:
                arc_angles = np.linspace(rang[0], rang[1], 50) + angle0
                arc_xs = 1 * (
                    pivot_point[0] + grasp_length * np.cos(arc_angles)
                )
                arc_ys = (
                    self.height
                    - 1
                    - 1 * (pivot_point[1] + grasp_length * np.sin(arc_angles))
                )
                pylab.plot(arc_xs, arc_ys, color=rgb, lw=4)

        cb = pylab.colorbar(
            pylab.cm.ScalarMappable(norm=norm, cmap=cmap),
            label="Number of Crossings Eliminated",
            orientation="vertical",
            ticks=bounds,
        )
        labels = np.arange(min_elim_num, max_elim_num + 1, 1)
        loc = labels + 0.5
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        pylab.savefig(f"cableGraphs/composite_with_action_space.png")
        pylab.show()

    def calc_cost_elim(
        self, theta, total_length, grasp_point_id, pivot_point_id, cableID
    ):
        """The cost is a weighted sum of
        1. Negative distance to other cables after the move
            (need a distance metric)
        2. Curvature at the pivot point after the move
        3. Uncertainty, which is negatively correlated with
            grasp_length/total_length (total length is the cable length from
            free end to pivot)
        """
        cost = 0
        graph: Graph = self.cg.graphs[cableID]
        # from res free end to res grasp to pivot to fixed end
        graph_this = self.simulate_next_state(
            theta, grasp_point_id, pivot_point_id, cableID
        )
        grasp_length = graph.compute_length(grasp_point_id, pivot_point_id)

        for cableid, graph in self.cg.graphs.items():
            if cableid != cableID:
                dist_cost = -graph_this.calc_distance_between_graphs(
                    self.cg.graphs[cableid]
                )
                cost += self.weight_dist * dist_cost
        curv_cost = -graph_this.calc_curvature(pivot_point_id)

        uncertainty_cost = -grasp_length / total_length
        cost = (
            cost / (len(self.cg.graphs) - 1)
            + self.weight_curv * curv_cost
            + self.weight_uncertainty * uncertainty_cost
        )

        return cost

    def calc_cost_redistrib(
        self, theta, total_length, grasp_point_id, pivot_point_id, cableID
    ):
        """The cost is a weighted sum of
        1. Cov of distance to upper boundary
        2. Uncertainty
        """
        cost = 0
        graph: Graph = self.cg.graphs[cableID]
        # from res free end to res grasp to pivot to fixed end
        graph_this = self.simulate_next_state(
            theta, grasp_point_id, pivot_point_id, cableID
        )
        grasp_length = graph.compute_length(grasp_point_id, pivot_point_id)
        # y coordinates of each node in the new graph
        dists = np.array(
            [
                graph_this.get_node_coords(node)[1]
                for node in graph_this.get_nodes()
            ]
        )
        dist_cov_cost = np.cov(dists)
        uncertainty_cost = -grasp_length / total_length

        cost = (
            self.weight_dist_cov * dist_cov_cost
            + self.weight_uncertainty * uncertainty_cost
        )
        return cost

    def search_goal_coord(self, nodes, keypoint_id, cableID, mode):
        # imagine pulling tight the cable segment from grasp point to pivot
        graph: Graph = self.cg.graphs[cableID]

        # the pivot point is the predecessor of the keypoint (an undercx)
        pivot_point_id = graph.get_pred(keypoint_id)
        theta_ranges_all = []  # [(gpid, theta_ranges), (..., ...)]
        total_max_elim_num = None
        for grasp_point_id in nodes:
            print("grasp: ", grasp_point_id, "pivot: ", pivot_point_id)

            grasp_length = graph.compute_length(grasp_point_id, pivot_point_id)

            # draw multiple arcs on a circle of R=grasp_length around pivot
            # this is the action space
            theta_ranges, max_elim_num = self.generate_action_space(
                grasp_point_id, pivot_point_id, grasp_length, cableID, vis=False
            )
            if theta_ranges is None:
                continue
            if mode == MODE_ELIM:  # in this mode we DO want to elim cx
                # get rid of the item in theta_ranges whose key (elim num) is 0
                theta_ranges.pop(0, None)
            print("theta ranges (filtered):", theta_ranges)
            if len(theta_ranges) == 0:
                continue
            if total_max_elim_num is None or max_elim_num > total_max_elim_num:
                total_max_elim_num = max_elim_num
            theta_ranges_all.append((grasp_point_id, theta_ranges))
        if len(theta_ranges_all) == 0:
            return None, None, None

        # to save space, we only select from the first 8 gpid
        theta_ranges_all = theta_ranges_all[:8]
        # gp_cost_map_all <- {gpid: {elim: cost, elim: cost}, ..}
        # note: if mode is REDISTRIB, theta_ranges_all only contain theta_ranges
        # whose key (the elim num) is 0
        gp_cost_map_all, gp_theta_map_all = self.compute_all_costs(
            theta_ranges_all,
            total_max_elim_num,
            pivot_point_id,
            cableID,
            mode,
            vis=True,
        )
        if not gp_cost_map_all:
            return None, None, None

        # select best action
        elim_to_best_cost_map = {}
        elim_to_best_action_map = {}  # {elim: (gpid, theta)}
        for gpid in gp_cost_map_all.keys():
            gp_elim_cost_map = gp_cost_map_all[gpid]
            gp_elim_theta_map = gp_theta_map_all[gpid]
            for elim, cost in gp_elim_cost_map.items():
                best_cost = elim_to_best_cost_map.get(elim)
                if best_cost is None or cost < best_cost:
                    elim_to_best_cost_map[elim] = cost
                    theta = gp_elim_theta_map[elim]
                    elim_to_best_action_map[elim] = (gpid, theta)

        best_elim = None

        if mode == MODE_ELIM:  # in elim cx mode
            best_score = None
            for elim, cost in elim_to_best_cost_map.items():
                score = -elim * self.weight_elim + cost * self.weight_cost
                if best_elim is None or score < best_score:
                    best_elim = elim
                    best_score = score

        elif mode == MODE_REDISTRIB:
            # if we're in redistribution mode, no gpid has any theta will result
            # in a reduction of #cx
            assert total_max_elim_num == 0
            best_elim = 0

        best_gpid, best_theta = elim_to_best_action_map[best_elim]
        best_grasp_length = graph.compute_length(best_gpid, pivot_point_id)
        # convert theta into goal coord
        goal_coord, goal_vec = self.theta_to_goal_coord(
            best_theta, best_gpid, pivot_point_id, best_grasp_length, cableID
        )
        return best_gpid, goal_coord, goal_vec

    def compute_all_costs(
        self,
        theta_ranges_all,
        total_max_elim_num,
        pivot_point_id,
        cableID,
        mode,
        vis=True,
    ):
        """Depending on which mode we're in, select a type of cost to use"""
        graph: Graph = self.cg.graphs[cableID]
        if vis is True:
            fig, axs = plt.subplots(1, total_max_elim_num + 1, squeeze=False)
            fig.suptitle("Cost in action subspace")
        cmap_disc = pylab.get_cmap("hsv", len(theta_ranges_all))

        cost_fn = (
            self.calc_cost_elim
            if mode == MODE_ELIM
            else self.calc_cost_redistrib
        )

        gp_cost_map_all = {}  # {gpid: {elim: cost, elim: cost}, ..}
        gp_theta_map_all = {}
        plotted_legend = set()
        for i, (gpid, theta_ranges) in enumerate(theta_ranges_all):
            gp_elim_cost_map = {}  # {elim: cost}
            gp_elim_theta_map = {}  # {elim: theta}
            for elim, theta_range in theta_ranges.items():
                axs[0, elim].set_title(f"#Cx Eliminated: {elim}")
                best_cost = None
                best_theta = None
                for subrange in theta_range:
                    # grasp_length = graph.compute_length(gpid, pivot_point_id)
                    total_length = graph.compute_length(
                        graph.get_free_endpoint(), pivot_point_id
                    )
                    thetas_ = np.linspace(subrange[0], subrange[1], num=20)
                    res = minimize_scalar(
                        cost_fn,
                        args=(total_length, gpid, pivot_point_id, cableID),
                        bounds=subrange,
                        method="bounded",
                    )
                    print(f"optimize in mode {mode}")
                    print(res)
                    all_costs = np.array(
                        [
                            cost_fn(
                                the, total_length, gpid, pivot_point_id, cableID
                            )
                            for the in thetas_
                        ]
                    )
                    if res.success is True:
                        if best_cost is None or res.fun < best_cost:
                            best_cost = res.fun
                            best_theta = res.x

                    rgb = cmap_disc(i)
                    if (elim, gpid) not in plotted_legend:
                        axs[0, elim].plot(
                            thetas_, all_costs, color=rgb, label=f"{gpid}"
                        )
                        plotted_legend.add((elim, gpid))
                    else:
                        axs[0, elim].plot(thetas_, all_costs, color=rgb)
                    # the optimal theta in each subrange
                    axs[0, elim].plot(res.x, res.fun, color="red", marker="o")
                if best_cost is not None:
                    gp_elim_cost_map[elim] = best_cost
                    gp_elim_theta_map[elim] = best_theta
                    axs[0, elim].plot(
                        best_theta, best_cost, color="green", marker="o"
                    )
            if gp_elim_cost_map:
                gp_cost_map_all[gpid] = gp_elim_cost_map
                gp_theta_map_all[gpid] = gp_elim_theta_map
        if gp_cost_map_all and vis is True:
            for ax in axs.flat:
                ax.set(xlabel=r"$\theta$", ylabel="Cost")
                ax.legend()
            plt.savefig(f"cableGraphs/cost.png")
            plt.show()
        return gp_cost_map_all, gp_theta_map_all

    def simulate_next_state(
        self, theta, grasp_point_id, pivot_point_id, cableID
    ):
        """Create the simulated graph (full) of a cable after an action. It
        tries to keep the newly grown edges to have similar length as the
        existing edges.
        TODO currently the modified nodes have default types"""
        graph: Graph = self.cg.graphs[cableID]
        grasp_length = graph.compute_length(grasp_point_id, pivot_point_id)
        total_length = graph.compute_length(
            graph.get_free_endpoint(), pivot_point_id
        )
        free_length = total_length - grasp_length
        pivot_point = np.array(graph.get_node_coords(pivot_point_id))
        # from the res grasp point (res point) to pivot, then to fixed endpoint
        graph_this = graph.build_subgraph(
            pivot_point_id, graph.get_fixed_endpoint()
        )
        grasp_point = np.array(graph.get_node_coords(grasp_point_id))
        free_endpoint = np.array(
            graph.get_node_coords(graph.get_free_endpoint())
        )
        # depending on the ratio of grasp_length/total_length, we
        # use either the straight mode or bending mode
        ratio = grasp_length / total_length
        res_grasp_point = self.get_result_point(
            theta, grasp_length, grasp_point, pivot_point
        )
        num_new_edges = self.calc_num_edges_in_branch(
            res_grasp_point, pivot_point, cableID
        )
        # grow one branch from res grasp point to pivot
        graph_this.grow_branch(
            res_grasp_point,
            pivot_point_id,
            div=num_new_edges,
            new_id=grasp_point_id,
        )
        if ratio > 0.7:  # use straight mode TODO magic number
            free_end_res_point = self.get_result_point(
                theta, total_length, grasp_point, pivot_point
            )
        else:  # use bending mode
            # the free-end tend to remain in its position
            res_grasp_point_to_orig_free_end_vec = unit_vector(
                free_endpoint - res_grasp_point
            )
            free_end_res_point = (
                res_grasp_point
                + res_grasp_point_to_orig_free_end_vec * free_length
            )
        # then grow one branch from res free end to res grasp point
        num_new_edges = self.calc_num_edges_in_branch(
            free_end_res_point, res_grasp_point, cableID
        )
        graph_this.grow_branch(
            free_end_res_point,
            grasp_point_id,
            div=num_new_edges,
            new_id=graph.get_free_endpoint(),
        )

        # add node types TODO add more?
        graph_this.add_free_endpoint(graph.get_free_endpoint())
        graph_this.add_fixed_endpoint(graph.get_fixed_endpoint())

        return graph_this

    def calc_num_edges_in_branch(self, res_point, pivot_point, cableID):
        graph: Graph = self.cg.graphs[cableID]
        # dist between res and pivot devided by average edge length in graph
        num_new_edges = np.round(
            calcDistance(
                pivot_point[0], pivot_point[1], res_point[0], res_point[1]
            )
            / (
                graph.compute_length(
                    graph.get_free_endpoint(), graph.get_fixed_endpoint()
                )
                / len(graph.get_edges())
            )
        ).astype(np.int32)
        num_new_edges = max(1, num_new_edges)
        return num_new_edges

    def theta_to_goal_coord(
        self, theta, grasp_point_id, pivot_point_id, length, cableID
    ):
        graph: Graph = self.cg.graphs[cableID]
        grasp_point = np.array(graph.get_node_coords(grasp_point_id))
        pivot_point = np.array(graph.get_node_coords(pivot_point_id))
        res_point = self.get_result_point(
            theta, length, grasp_point, pivot_point
        )
        goal_coord = np.floor(res_point).astype(np.int32)
        goal_vec = unit_vector(pivot_point - goal_coord)
        return goal_coord.tolist(), goal_vec.tolist()

    def is_bad_3d_coord(self, point_c):
        if (
            point_c[2] < 0.05 or point_c[2] > 1
        ):  # filter out points with wrong depth
            print("Bad 2d coord, no valid 3D coordinate!")
            return True
        return False

    def unweave_step(self, bgr, depth):
        """Attempt to either eliminate crossings or redistribute cables"""
        self.gen_graph_from_image(bgr)
        self.depth = depth
        num_done = 0
        print("Attemping to ELIM")
        for cableID in self.cg.graphs.keys():
            action = self.eliminate_crossing(cableID)
            if num_done > 0 and action is not None and action.is_empty is False:
                num_done = 0
            if action is not None:
                if action.is_empty is True:
                    # if this cable is done, we check other cables and see
                    # if we can perform a non-empty action
                    num_done += 1
                else:
                    print("Found valid action to elim at least 1 cx")
                    self.fa.exe_action(action)
                    return UNWEAVE_IN_PROGRESS

        if num_done == len(self.cg.graphs.keys()):
            print("Unweaving all done. Stopping...")
            return UNWEAVE_ALL_DONE
        # if cable not done and we can't find any action to elim, the we attemp
        # to redistrib a (random?) cable whose first cx is an ux
        print("Cannot find any move on cables whose first cx is OX")
        print("Attemping to REDISTRIB")
        cableids = list(self.cg.graphs.keys())
        random.shuffle(cableids)
        for cableID in cableids:
            action = self.redistribute_cable(cableID)
            if action is not None:
                print("Found valid action to redistrib a cable")
                self.fa.exe_action(action)
                return UNWEAVE_IN_PROGRESS
        # can neither ELIM nor REDISTRIB
        return UNWEAVE_FAIL

    def run(self):
        if self.use_rs is False or self.use_fa is False:
            print("Realsense or Franka Arm not in use, returning")
            return
        self.fa.reset_joint_and_gripper()
        self.fa.open_gripper()
        while True:
            self.fa.goto_capture_pose()
            vals = self.realsense.getFrameSet(skip_frames=5)
            if vals is None:
                raise RuntimeError("Failed to get frameset")
            depth, bgr = vals
            img_w = bgr.shape[1]
            img_h = bgr.shape[0]
            self.width = img_w
            self.height = img_h
            self.workspace = [(0, 0), (self.width, self.height)]
            cv2.imshow("img", bgr)
            cv2.waitKey(0)
            cv2.imwrite("cableImages/rs_cable_imgs2/test.png", bgr)
            res = self.unweave_step(bgr, depth)
            if res is UNWEAVE_FAIL:
                raise RuntimeError("Cannot find an action to unweave!")
            elif res is UNWEAVE_ALL_DONE:
                break
            reset_id()
            self.cg = CableGraph()
        print("Done unweaving all cables")
        self.realsense.close()


# test
def test_elim():
    img = cv2.imread("cableImages/generated_03.png")
    img_w = img.shape[1]
    img_h = img.shape[0]
    csp = CableSimplePolicy(
        width=img_w, height=img_h, use_fa=False, use_rs=False
    )
    csp.gen_graph_from_image(img)
    action = csp.eliminate_crossing("cable_red")
    action.print()


def test_redistrib():
    img = cv2.imread("cableImages/generated_03.png")
    img_w = img.shape[1]
    img_h = img.shape[0]
    csp = CableSimplePolicy(
        width=img_w, height=img_h, use_fa=False, use_rs=False
    )
    csp.gen_graph_from_image(img)
    action = csp.redistribute_cable("cable_blue")
    action.print()


if __name__ == "__main__":
    csp = CableSimplePolicy(use_fa=True, use_rs=True)
    csp.run()
    # test_elim()
