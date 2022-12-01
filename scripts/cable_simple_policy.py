import numpy as np
from graph_builder import Graph, CableGraph
from graph_builder import POS_DOWN, POS_UP, POS_NONE, NODE_FREE
from cable_discretization import Discretize
from action import Action
from my_franka import MyFranka
from rs_driver import Realsense
from utility import get_rotation_matrix, unit_vector, calcDistance
from scipy.optimize import minimize_scalar


class CableSimplePolicy:
    def __init__(self, width, height, use_rs=False):
        self.cg = CableGraph()
        self.disc = Discretize()
        self.fa = MyFranka()
        self.workspace = [(0, 0), (width, height)]  # top left, bot right
        self.weight_dist = 1.0
        self.weight_curv = 1.0
        self.use_rs = use_rs
        if self.use_rs is True:
            self.realsense = Realsense()

    def gen_graph_from_image(self, img):
        cable_data = self.disc.getCablesDataFromImage(img)
        self.cg.create_graphs(cable_data)
        self.cg.create_compound_graph()

    def eliminate_crossing(self, cableID):
        """Attempt to eliminate the first crossing of a cable.

        Return the action if exists, none o/w
        """
        graph: Graph = self.cg[cableID]

        next_id, cx_pos, nodes = graph.get_next_keypoint(
            graph.get_free_endpoint()
        )
        if len(nodes) == 0 or cx_pos == POS_DOWN:  # first crossing is undercx
            return None
        if graph.is_fixed_endpoint(next_id):  # this cable is already done
            return Action(False)
        next_id, nodes = graph.get_next_fixed_keypoint(
            graph.get_free_endpoint()
        )
        for node in nodes:
            # attemp to move this node to free space
            goal_coord = self.search_goal_coord(node, next_id, cableID)
            if goal_coord is not None:
                # TODO fill in action params
                return Action(True)
        return None

    def get_result_point(
        self, theta, length, grasp_point: np.ndarray, pivot_point: np.ndarray
    ) -> np.ndarray:
        zero_vec = unit_vector(grasp_point - pivot_point)
        # rotate zero_vec about the pivot point by theta
        rotated_vec = np.multiply(
            get_rotation_matrix(theta), zero_vec.reshape((-1, 1)).flatten()
        )
        res_point = pivot_point + rotated_vec * length
        return res_point  # not rounded, dtype is float

    def is_in_workspace(self, point):
        return (
            point[0] >= self.workspace[0][0]
            and point[0] < self.workspace[1][0]
            and point[1] >= self.workspace[0][1]
            and point[1] < self.workspace[1][1]
        )

    def get_num_crossings(self, pivot_point_id, res_point: list, cableID):
        """Build two tmp graphs and compute #cx by checking edge intersects"""
        graph: Graph = self.cg.graphs[cableID]
        fixed_endpoint_id = graph.get_fixed_endpoint()
        # from res point to pivot point then to fixed endpoint
        graph_this = graph.build_subgraph(pivot_point_id, fixed_endpoint_id)
        id_new = graph_this.add_node(NODE_FREE, coords=res_point)
        graph_this.add_edge(id_new, pivot_point_id, POS_NONE)

        graph_others = self.cg.create_compound_graph_except(cableID)

        return graph_this.get_num_crossings_two_graphs(graph_others)

    def generate_action_space(
        self, grasp_point_id, pivot_point_id, length, cableID
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
        thetas = np.linspace(-np.pi / 2, np.pi / 2, num=200)
        theta_ranges = {}  # {elim_num1: [[],[],..], elim_num2: [[],[],..]}
        theta_range_tmp = []
        elim_num = 0
        for theta in thetas:
            res_point = self.get_result_point(
                theta, length, grasp_point, pivot_point
            )
            res_ok = False
            # check whether the action is within workspace
            if self.is_in_workspace(res_point):
                # check #cx
                num_cx_orig = self.cg.compound_graph.get_num_crossings()
                num_cx_new = self.get_num_crossings(
                    pivot_point_id, res_point.tolist(), cableID
                )
                # minus one because the fixed endpoint was counted as a crossing
                # in num_cx_new
                tmp_elim_num = num_cx_orig - (num_cx_new - 1)
                if tmp_elim_num > 0:
                    res_ok = True
                    elim_num = tmp_elim_num
                    if len(theta_range_tmp) < 2:
                        theta_range_tmp.append(theta)
                    else:
                        theta_range_tmp[1] = theta
            # reset theta range and elim num
            if res_ok is False:
                if len(theta_range_tmp) == 2:
                    if theta_ranges.get(elim_num) is None:
                        theta_ranges[elim_num] = []
                    theta_ranges[elim_num].append(theta_range_tmp)
                theta_range_tmp = []
                elim_num = 0
        # only return the theta ranges with the biggest elim num
        max_elim_num = np.max(list(theta_ranges.keys()))
        return theta_ranges[max_elim_num]

    def calc_cost(self, theta, length, grasp_point_id, pivot_point_id, cableID):
        """The cost is a weighted sum of
        1. Negative distance to other cables after the move
            (need a distance metric)
        2. Curvature at the pivot point after the move
        """
        cost = 0
        graph: Graph = self.cg.graphs[cableID]
        pivot_point = np.array(graph.get_node_coords(pivot_point_id))
        # from theoritical endpoint (res point) to pivot, then to fixed endpoint
        graph_this = graph.build_subgraph(
            pivot_point_id, graph.get_fixed_endpoint()
        )
        grasp_point = np.array(graph.get_node_coords(grasp_point_id))
        res_point = self.get_result_point(
            theta, length, grasp_point, pivot_point
        )
        # dist between res and pivot devided by average edge length in graph
        num_new_edges = calcDistance(
            pivot_point[0], pivot_point[1], res_point[0], res_point[1]
        ) / (
            graph.compute_length(
                graph.get_free_endpoint(), graph.get_fixed_endpoint()
            )
            / len(graph.get_edges())
        )
        graph_this.grow_branch(res_point, pivot_point_id, div=num_new_edges)

        for cableid, graph in self.cg.graphs.items():
            if cableid != cableID:
                dist_cost = -graph_this.calc_distance_between_graphs(
                    self.cg.graphs[cableid]
                )
                curv_cost = graph_this.calc_curvature(pivot_point_id)
                cost += (
                    self.weight_dist * dist_cost + self.weight_curv * curv_cost
                )
        return cost

    def search_goal_coord(self, grasp_point_id, keypoint_id, cableID):
        # imagine pulling tight the cable segment from grasp point to pivot
        graph: Graph = self.cg.graphs[cableID]
        # the pivot point is the predecessor of the keypoint (an undercx)
        pivot_point_id = graph.get_pred(keypoint_id)
        length = graph.compute_length(grasp_point_id, pivot_point_id)

        # draw a circle (or multiple arcs on a circle) around pivot point.
        # this is the action space
        theta_ranges = self.generate_action_space(
            grasp_point_id, pivot_point_id, length, cableID
        )
        thetas = []
        costs = []
        for theta_range in theta_ranges:
            res = minimize_scalar(
                self.calc_cost,
                args=(length, pivot_point_id, cableID),
                bounds=theta_range,
                method="bounded",
            )
            if res.status is True:
                thetas.append(res.x)
                costs.append(res.fun)
        theta = thetas[np.argmin(costs)]
        return theta

    def unweave_step(self, bgr, depth):
        self.gen_graph_from_image(bgr)
        for cableID in self.cg.graphs.keys():
            action = self.eliminate_crossing(cableID)
            if action is not None:
                if action.is_empty is False:
                    # TODO add 3d info to the action using depth
                    action.pick_3d = [0, 0, 0]
                    action.place_3d = [0, 0, 0]
                    self.fa.exe_action(action)
                return True
        return False

    def run(self):
        if self.use_rs is False:
            print("Realsense not in use, returning")
            return
        self.fa.reset_joint_and_gripper()
        self.fa.goto_capture_pose()
        while len(self.cg.compound_graph.get_crossings()) > 0:
            vals = self.realsense.getFrameSet(skip_frames=5)
            if vals is None:
                raise RuntimeError("Failed to get frameset")
            depth, bgr = vals
            if self.unweave_step(bgr, depth) is False:
                raise RuntimeError("Cannot find an action to unweave!")
        print("Done unweaving all cables")
