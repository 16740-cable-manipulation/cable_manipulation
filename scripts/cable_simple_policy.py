import numpy as np
from graph_builder import Graph, CableGraph
from graph_builder import POS_DOWN, POS_UP, POS_NONE, NODE_FREE
from cable_discretization import Discretize
from action import Action
from my_franka import MyFranka
from rs_driver import Realsense
from utility import get_rotation_matrix, unit_vector


class CableSimplePolicy:
    def __init__(self, width, height, use_rs=False):
        self.cg = CableGraph()
        self.disc = Discretize()
        self.fa = MyFranka()
        self.workspace = [(0, 0), (width, height)]  # top left, bot right
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
    ):
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
        theta_ranges = []
        theta_range_tmp = []
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
                    pivot_point_id, res_point, cableID
                )
                # minus one because the fixed endpoint was counted as a crossing
                # in num_cx_new
                if num_cx_orig > num_cx_new - 1:
                    res_ok = True
                    if len(theta_range_tmp) < 2:
                        theta_range_tmp.append(theta)
                    else:
                        theta_range_tmp[1] = theta
            if res_ok is False:
                if len(theta_range_tmp) == 2:
                    theta_ranges.append(theta_range_tmp)
                theta_range_tmp = []
        return theta_ranges

    def calc_cost(self, theta, length, pivot_point, cableID):
        """The cost is a weighted sum of 
        1. Negative distance to other cables after the move 
            (need a distance metric)
        2. Curvature at the pivot point after the move
        """
        return 0

    def grad_dist(self, theta, length, pivot_point, cableID):
        return 0

    def grad_curv(self, theta, length, pivot_point, cableID):
        return 0

    def optimize_theta(self, theta_range, length, pivot_point, cableID):
        alpha = 0.1  # step size
        max_iter = 5000
        eps = 1e-4
        iter = 0
        theta = np.random.uniform(low=theta_range[0], high=theta_range[1])
        while iter < max_iter:
            grad_total = 0
            grad_dist = self.grad_dist(theta, length, pivot_point, cableID)
            grad_curv = self.grad_curv(theta, length, pivot_point, cableID)
            grad_total = 0.5 * grad_dist + 0.5 * grad_curv
            if grad_total < eps:
                break
            iter = iter + 1
            theta = theta - alpha * grad_total
        cost = self.calc_cost(theta, length, pivot_point, cableID)
        return theta, cost

    def search_goal_coord(self, grasp_point_id, pivot_point_id, cableID):
        # imagine pulling tight the cable segment from grasp point to pivot
        graph: Graph = self.cg.graphs[cableID]
        length = graph.compute_length(grasp_point_id, pivot_point_id)

        # draw a circle (or multiple arcs on a circle) around pivot point.
        # this is the action space
        theta_ranges = self.generate_action_space(
            grasp_point_id, pivot_point_id, length, cableID
        )
        thetas = []
        costs = []
        for theta_range in theta_ranges:
            theta, cost = self.optimize_theta(
                theta_range, length, pivot_point_id, cableID
            )
            thetas.append(theta)
            costs.append(cost)
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
