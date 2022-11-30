import numpy as np
from graph_builder import Graph, CableGraph
from graph_builder import POS_DOWN, POS_UP, POS_NONE
from cable_discretization import Discretize
from action import Action
from my_franka import MyFranka
from rs_driver import Realsense


class CableSimplePolicy:
    def __init__(self, use_rs=False):
        self.cg = CableGraph()
        self.disc = Discretize()
        self.fa = MyFranka()
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

    def generate_action_space(self, pivot_point, length, cableID):
        """Return a np array [[th_start1, th_end1], [th_start2, th_end2],..]
        The actions in the action space should theoretically eliminate at least
        one cx"""
        return

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

    def search_goal_coord(self, grasp_point, pivot_point, cableID):
        # imagine pulling tight the cable segment from grasp point to pivot
        graph: Graph = self.cg.graphs[cableID]
        length = graph.compute_length(grasp_point, pivot_point)

        # draw a circle (or multiple arcs on a circle) around pivot point.
        # this is the action space
        theta_ranges = self.generate_action_space(pivot_point, length, cableID)
        thetas = []
        costs = []
        for theta_range in theta_ranges:
            theta, cost = self.optimize_theta(
                theta_range, length, pivot_point, cableID
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
                quit(1)
            depth, bgr = vals
            if self.unweave_step(bgr, depth) is False:
                raise RuntimeError("Cannot perform unweave step")
                quit(1)
        print("Done unweaving all cables")
