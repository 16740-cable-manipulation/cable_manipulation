from tkinter import N
from tkinter.messagebox import NO
import numpy as np
import cv2
from graph_builder import Graph, CableGraph, reset_id
from graph_builder import POS_DOWN, POS_UP, POS_NONE, NODE_FREE, DPI
from cable_discretization import getCablesDataFromImage
from action import Action

# from my_franka import MyFranka
from rs_driver import Realsense
from utility import get_rotation_matrix, unit_vector, calcDistance
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pylab as pylab

UNWEAVE_IN_PROGRESS = 0
UNWEAVE_ALL_DONE = 1
UNWEAVE_FAIL = 2


class CableSimplePolicy:
    def __init__(self, width=640, height=480, use_fa=False, use_rs=False):
        self.cg = CableGraph()
        self.use_fa = use_fa
        if use_fa:
            self.fa = MyFranka()
        self.width = width
        self.height = height
        self.workspace = [
            (0, 0),
            (self.width, self.height),
        ]  # top left, bot right
        self.depth = None
        self.weight_dist = 1.0
        self.weight_curv = 40
        self.weight_uncertainty = 100

        self.weight_cost = 1.0
        self.weight_elim = 30

        self.rim = 30
        self.min_lift_z = 0.06
        self.max_lift_z = 0.3
        self.use_rs = use_rs
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
        if len(nodes) == 0 or cx_pos == POS_DOWN:  # first crossing is undercx
            print("First cx is undercx. ", cableID, " not movable")
            return None
        if graph.is_fixed_endpoint(next_id):  # this cable is already done
            print("First cx is fixed endpoint. ", cableID, " already done")
            return Action(False)
        print(f"cable # {cableID} is movable")
        # test get next keypoint
        next_id, nodes = graph.get_next_fixed_keypoint(
            graph.get_free_endpoint()
        )
        nodes = nodes[:-1]  # the last one is the pivot point
        # for node in nodes:
        # attemp to move this node to free space
        grasp_point_id, goal_coord, goal_vec = self.search_goal_coord(
            nodes, next_id, cableID
        )
        print("goal: ", grasp_point_id, goal_coord, goal_vec)
        if goal_coord is not None:
            # fill in action params (2d, except z)
            action = Action(True)
            pick_point = graph.get_node_coords(grasp_point_id)
            action.pick_coord = pick_point
            action.place_coord = goal_coord
            pivot_point = graph.get_node_coords(next_id)
            px_dist_pivot_pick = calcDistance(
                pick_point[0], pick_point[1], pivot_point[0], pivot_point[1]
            )
            px_dist_pivot_place = calcDistance(
                goal_coord[0], goal_coord[1], pivot_point[0], pivot_point[1]
            )
            tmp_px_dist = np.sqrt(
                px_dist_pivot_place**2 - px_dist_pivot_pick**2
            )
            action.z = np.clip(
                self.px_length_to_m(cableID, tmp_px_dist) * 0.8,
                self.min_lift_z,
                self.max_lift_z,
            )
            # self.z_mult * graph.compute_length(node, next_id)
            # the direction vector is tangent to cable at grasp point
            action.pick_vec = graph.calc_tangent_vec(grasp_point_id)
            action.place_vec = goal_vec
            return action
        print("Could not find an action to eliminate cx on this cable")
        return None

    def get_cable_segment_length(self, cableID, pivot_point, goal_coord):
        len_px = calcDistance(
            pivot_point[0], pivot_point[1], goal_coord[0], goal_coord[1]
        )
        return self.px_length_to_m(cableID, len_px)

    def px_length_to_m(self, cableID, len_px):
        # get a random edge
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

    def get_num_crossings(
        self,
        pivot_point_id,
        grasp_point_id,
        res_point: list,
        cableID,
        vis=False,
        save=False,
    ):
        """Build two tmp graphs and compute #cx by checking edge intersects
        Will not count the fixed end point
        """
        graph_this = self.generate_transitioned_graph(
            pivot_point_id, grasp_point_id, res_point, cableID
        )
        graph_others: Graph = self.cg.create_compound_graph_except(cableID)

        save_path = f"cableGraphs/numcx_composite_transitioned.png"
        if vis is True:
            composite_graph = graph_this.compose(graph_others)
            if save is True:
                composite_graph.visualize(save_path=save_path)
            else:
                composite_graph.visualize()
        line_seg_pt1 = graph_this.get_node_coords(grasp_point_id)
        line_seg_pt2 = graph_this.get_node_coords(pivot_point_id)
        num_cx = graph_others.get_num_crossings_with_line_seg(
            line_seg_pt1, line_seg_pt2
        )
        # also subtract repeated count at the fixed endpoint
        return num_cx

    def generate_transitioned_graph(
        self, pivot_point_id, grasp_point_id, res_point: list, cableID
    ) -> Graph:
        graph: Graph = self.cg.graphs[cableID]
        fixed_endpoint_id = graph.get_fixed_endpoint()
        # from res point to pivot point then to fixed endpoint
        graph_this = graph.build_subgraph(pivot_point_id, fixed_endpoint_id)
        graph_this.add_node_id(grasp_point_id, NODE_FREE, coords=res_point)
        graph_this.add_edge(grasp_point_id, pivot_point_id, POS_NONE)
        return graph_this

    def generate_action_space(
        self,
        grasp_point_id,
        pivot_point_id,
        grasp_length,
        total_length,
        cableID,
    ):
        """Return a list [[th_start1, th_end1], [th_start2, th_end2],..]
        The actions in the action space should theoretically eliminate at least
        one cx. Also, it shouldn't exceed the workspace"""
        # assume we limit the theta to -pi/2 ~ pi/2 deg
        # note: positive theta is clockwise (cuz it's in cv coord)
        # theta=0 is the line connecting pivot point to grasp point

        graph: Graph = self.cg.graphs[cableID]
        grasp_point = np.array(graph.get_node_coords(grasp_point_id))
        pivot_point = np.array(graph.get_node_coords(pivot_point_id))
        thetas = np.linspace(-np.pi / 2, np.pi / 2, num=100)
        thetas = np.hstack(([0], thetas))
        theta_ranges = {}  # {elim_num1: [[],[],..], elim_num2: [[],[],..]}
        theta_range_tmp = []
        elim_num = 0

        for i, theta in enumerate(thetas):
            # res_point is where we might place the grasped point
            res_point = self.get_result_point(
                theta, grasp_length, grasp_point, pivot_point
            )
            res_ok = False
            # check whether the res point is within workspace
            if self.is_in_workspace(res_point):
                # check #cx
                # assuming that after grasping somewhere on the movable part
                # of the cable, the movable part automatically straightens
                res_point_free_endpoint = self.get_result_point(
                    theta, total_length, grasp_point, pivot_point
                )
                # subgraph from free ep to pivot
                graph_orig_sub = graph.build_subgraph(
                    graph.get_free_endpoint(), pivot_point_id
                )
                num_cx_orig = graph_orig_sub.get_num_crossings()
                num_cx_new = self.get_num_crossings(
                    pivot_point_id,
                    grasp_point_id,
                    res_point_free_endpoint.tolist(),
                    cableID,
                )
                tmp_elim_num = num_cx_orig - num_cx_new
                if tmp_elim_num > 0:
                    res_ok = True
                    if tmp_elim_num != elim_num:  # elim num changed
                        if len(theta_range_tmp) == 2:
                            if theta_ranges.get(elim_num) is None:
                                theta_ranges[elim_num] = []
                            theta_ranges[elim_num].append(theta_range_tmp)
                        elif len(theta_range_tmp) == 1:
                            if theta_ranges.get(elim_num) is None:
                                theta_ranges[elim_num] = []
                            theta_range_tmp.append(theta)
                            theta_ranges[elim_num].append(theta_range_tmp)
                        theta_range_tmp = []

                    elim_num = tmp_elim_num
                    if len(theta_range_tmp) < 2:
                        theta_range_tmp.append(theta)
                    else:
                        theta_range_tmp[1] = theta
                # debug
                print(
                    "theta: ",
                    theta,
                    " num_cx_new: ",
                    num_cx_new,
                    " num_cx_orig: ",
                    num_cx_orig,
                    " res_ok: ",
                    res_ok,
                )
                # _ = self.get_num_crossings(
                #     pivot_point_id,
                #     grasp_point_id,
                #     res_point.tolist(),
                #     cableID,
                #     vis=False,
                #     save=False,
                # )
            # for plotting
            if i == 0:
                res_ok = False
                theta_range_tmp = []
                elim_num = 0
                # plot the zero theta and save
                _ = self.get_num_crossings(
                    pivot_point_id,
                    grasp_point_id,
                    res_point.tolist(),
                    cableID,
                    vis=True,
                    save=True,
                )
                continue
            # reset theta range and elim num
            if res_ok is False or i == thetas.shape[0] - 1:
                if len(theta_range_tmp) == 2:
                    if theta_ranges.get(elim_num) is None:
                        theta_ranges[elim_num] = []
                    theta_ranges[elim_num].append(theta_range_tmp)
                theta_range_tmp = []
                elim_num = 0

        if not theta_ranges:
            return None, None
        # only return the theta ranges with the biggest elim num
        max_elim_num = np.max(list(theta_ranges.keys()))
        # draw on fig the theta ranges
        angle0 = self.get_zero_theta_vector_angle(grasp_point, pivot_point)
        # for plotting
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
        max_elim_num = np.max(list(theta_ranges.keys())) + 2
        min_elim_num = 1
        # cmap = plt.get_cmap("cool", max_elim_num - min_elim_num + 1)

        fig, ax = pylab.subplots(
            1, 1, figsize=(self.width / DPI, self.height / DPI), dpi=DPI
        )

        # pylab.imshow(mpimg.imread(save_path))
        graph: Graph = self.cg.graphs[cableID]
        zero_res_point = self.get_result_point(
            0.0,
            grasp_length,
            np.array(graph.get_node_coords(grasp_point_id)),
            pivot_point,
        )
        graph_this = self.generate_transitioned_graph(
            pivot_point_id, grasp_point_id, zero_res_point.tolist(), cableID
        )
        graph_others: Graph = self.cg.create_compound_graph_except(cableID)
        composite_graph = graph_this.compose(graph_others)
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
            rgb = cmap_disc(elim - 1)
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
            label=r"|$V_{elim}$|",
            orientation="vertical",
            ticks=bounds,
        )
        labels = np.arange(min_elim_num, max_elim_num + 1, 1)
        loc = labels + 0.5
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        pylab.savefig(f"cableGraphs/composite_with_action_space.png")
        pylab.show()

    def calc_cost(self, theta, length, grasp_point_id, pivot_point_id, cableID):
        """The cost is a weighted sum of
        1. Negative distance to other cables after the move
            (need a distance metric)
        2. Curvature at the pivot point after the move
        3. Uncertainty, which is negatively correlated with grasp_length/total_length
        """
        cost = 0
        graph: Graph = self.cg.graphs[cableID]
        pivot_point = np.array(graph.get_node_coords(pivot_point_id))
        # from theoritical endpoint (res point) to pivot, then to fixed endpoint
        graph_this = graph.build_subgraph(
            pivot_point_id, graph.get_fixed_endpoint()
        )
        grasp_point = np.array(graph.get_node_coords(grasp_point_id))
        grasp_length = calcDistance(
            grasp_point[0], grasp_point[1], pivot_point[0], pivot_point[1]
        )
        res_point = self.get_result_point(
            theta, length, grasp_point, pivot_point
        )
        # dist between res and pivot devided by average edge length in graph
        num_new_edges = np.ceil(
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
        graph_this.grow_branch(res_point, pivot_point_id, div=num_new_edges)

        for cableid, graph in self.cg.graphs.items():
            if cableid != cableID:
                dist_cost = -graph_this.calc_distance_between_graphs(
                    self.cg.graphs[cableid]
                )
                curv_cost = -graph_this.calc_curvature(pivot_point_id)
                uncertainty_cost = -grasp_length / length
                cost += (
                    self.weight_dist * dist_cost
                    + self.weight_curv * curv_cost
                    + self.weight_uncertainty * uncertainty_cost
                )
        return cost

    def search_goal_coord(self, nodes, keypoint_id, cableID):
        # imagine pulling tight the cable segment from grasp point to pivot
        graph: Graph = self.cg.graphs[cableID]

        # the pivot point is the predecessor of the keypoint (an undercx)
        pivot_point_id = graph.get_pred(keypoint_id)
        theta_ranges_all = []  # [(gpid, theta_ranges), (..., ...)]
        total_max_elim_num = None
        for grasp_point_id in nodes:
            print("grasp: ", grasp_point_id, "pivot: ", pivot_point_id)

            grasp_length = graph.compute_length(grasp_point_id, pivot_point_id)
            total_length = graph.compute_length(
                graph.get_free_endpoint(), pivot_point_id
            )

            # draw a circle (or multiple arcs on a circle) around pivot point.
            # this is the action space
            theta_ranges, max_elim_num = self.generate_action_space(
                grasp_point_id,
                pivot_point_id,
                grasp_length,
                total_length,
                cableID,
            )
            if theta_ranges is None:
                continue
            if total_max_elim_num is None or max_elim_num > total_max_elim_num:
                total_max_elim_num = max_elim_num
            print(theta_ranges)
            theta_ranges_all.append((grasp_point_id, theta_ranges))
        if len(theta_ranges_all) == 0:
            return None, None, None

        # to save space, we only select from the first 8 gpid
        theta_ranges_all = theta_ranges_all[:8]
        fig, axs = plt.subplots(1, total_max_elim_num, squeeze=False)
        fig.suptitle("Cost in action subspace")
        cmap_disc = pylab.get_cmap("hsv", len(theta_ranges_all))

        gp_cost_map_all = {}  # {gpid: {elim: cost, elim: cost}, ..}
        gp_theta_map_all = {}
        for i, (gpid, theta_ranges) in enumerate(theta_ranges_all):
            gp_elim_cost_map = {}  # {elim: cost}
            gp_elim_theta_map = {}  # {elim: theta}
            for elim, theta_range in theta_ranges.items():
                best_cost = None
                best_theta = None
                for subrange in theta_range:
                    grasp_length = graph.compute_length(
                        grasp_point_id, pivot_point_id
                    )
                    total_length = graph.compute_length(
                        graph.get_free_endpoint(), pivot_point_id
                    )
                    res = minimize_scalar(
                        self.calc_cost,
                        args=(
                            total_length,
                            grasp_point_id,
                            pivot_point_id,
                            cableID,
                        ),
                        bounds=subrange,
                        method="bounded",
                    )
                    print(res)
                    if res.success is True:
                        if best_cost is None or res.fun < best_cost:
                            best_cost = res.fun
                            best_theta = res.x
                    thetas_ = np.linspace(subrange[0], subrange[1], num=60)
                    all_costs = np.array(
                        [
                            self.calc_cost(
                                the,
                                total_length,
                                grasp_point_id,
                                pivot_point_id,
                                cableID,
                            )
                            for the in thetas_
                        ]
                    )
                    rgb = cmap_disc(i)
                    axs[0, elim - 1].plot(thetas_, all_costs, color=rgb)
                    # the optimal theta in each subrange
                    axs[0, elim - 1].plot(
                        res.x, res.fun, color="red", marker="o"
                    )
                if best_cost is not None:
                    gp_elim_cost_map[elim] = best_cost
                    gp_elim_theta_map[elim] = best_theta
                    axs[0, elim - 1].plot(
                        best_theta, best_cost, color="green", marker="o"
                    )
            if gp_elim_cost_map:
                gp_cost_map_all[gpid] = gp_elim_cost_map
                gp_theta_map_all[gpid] = gp_elim_theta_map

        if not gp_cost_map_all:
            return None, None, None
        for ax in axs.flat:
            ax.set(xlabel=r"$\theta$", ylabel="Cost")
        plt.savefig(f"cableGraphs/cost.png")
        plt.show()

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
        best_score = None
        for elim, cost in elim_to_best_cost_map.items():
            score = elim * self.weight_elim + cost * self.weight_cost
            if best_elim is None or score < best_score:
                best_elim = elim
                best_score = score
        best_gpid, best_theta = elim_to_best_action_map[best_elim]
        best_grasp_length = graph.compute_length(best_gpid, pivot_point_id)
        # convert theta into goal coord
        goal_coord, goal_vec = self.theta_to_goal_coord(
            best_theta, best_gpid, pivot_point_id, best_grasp_length, cableID
        )
        return best_gpid, goal_coord, goal_vec

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
        self.gen_graph_from_image(bgr)
        self.depth = depth
        num_done = 0
        for cableID in self.cg.graphs.keys():
            action = self.eliminate_crossing(cableID)
            if num_done > 0 and action is not None and action.is_empty is False:
                num_done = 0
            if action is not None:
                if action.is_empty is True:
                    num_done += 1
                else:
                    print("Found valid action")
                    pick_point_3d_c = self.realsense.deproject_pixel(
                        depth, action.pick_coord[0], action.pick_coord[1]
                    )
                    place_point_3d_c = self.realsense.deproject_pixel(
                        depth, action.place_coord[0], action.place_coord[1]
                    )
                    if (
                        self.is_bad_3d_coord(pick_point_3d_c) is True
                        or self.is_bad_3d_coord(place_point_3d_c) is True
                    ):
                        continue
                    action.pick_3d = pick_point_3d_c
                    action.place_3d = place_point_3d_c
                    # for the direction vector, directly use the 2d vector
                    action.pick_vec_3d = np.array(
                        [action.pick_vec[0], action.pick_vec[1], 0]
                    )
                    action.place_vec_3d = np.array(
                        [action.place_vec[0], action.place_vec[1], 0]
                    )
                    self.fa.exe_action(action)
                    return UNWEAVE_IN_PROGRESS
        if num_done == len(self.cg.graphs.keys()):
            print("Unweaving all done. Stopping...")
            return UNWEAVE_ALL_DONE
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
def test_simple_policy():
    img = cv2.imread("cableImages/generated_03.png")
    img_w = img.shape[1]
    img_h = img.shape[0]
    csp = CableSimplePolicy(
        width=img_w, height=img_h, use_fa=False, use_rs=False
    )
    csp.gen_graph_from_image(img)
    action = csp.eliminate_crossing("cable_red")
    action.print()


if __name__ == "__main__":
    # csp = CableSimplePolicy(use_fa=True, use_rs=True)
    # csp.run()
    test_simple_policy()
