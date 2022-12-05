from tkinter import N
import numpy as np
import cv2
from graph_builder import Graph, CableGraph
from graph_builder import POS_DOWN, POS_UP, POS_NONE, NODE_FREE
from cable_discretization import getCablesDataFromImage
from action import Action

from my_franka import MyFranka
from rs_driver import Realsense
from utility import get_rotation_matrix, unit_vector, calcDistance
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib as mpl

UNWEAVE_IN_PROGRESS = 0
UNWEAVE_ALL_DONE = 1
UNWEAVE_FAIL = 2


class CableSimplePolicy:
    def __init__(self, width=640, height=480, use_rs=False):
        self.cg = CableGraph()
        self.fa = MyFranka()
        self.width = width
        self.height = height
        self.workspace = [
            (0, 0),
            (self.width, self.height),
        ]  # top left, bot right
        self.weight_dist = 1.0
        self.weight_curv = 60
        self.rim = 40
        self.lift_z = 0.06
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
        for node in nodes:
            # attemp to move this node to free space
            goal_coord, goal_vec = self.search_goal_coord(
                node, next_id, cableID
            )
            print("goal: ", goal_coord, goal_vec)
            if goal_coord is None:
                continue

            if goal_coord is not None:
                # fill in action params (2d, except z)
                action = Action(True)
                action.pick_coord = graph.get_node_coords(node)
                action.place_coord = goal_coord
                action.z = self.lift_z  # TODO should change with grasp length
                # self.z_mult * graph.compute_length(node, next_id)
                # the direction vector is tangent to cable at grasp point
                action.pick_vec = graph.calc_tangent_vec(node)
                action.place_vec = goal_vec
                return action
        print("Could not find an action to eliminate cx on this cable")
        return None

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
        graph: Graph = self.cg.graphs[cableID]
        fixed_endpoint_id = graph.get_fixed_endpoint()
        # from res point to pivot point then to fixed endpoint
        graph_this = graph.build_subgraph(pivot_point_id, fixed_endpoint_id)
        graph_this.add_node_id(grasp_point_id, NODE_FREE, coords=res_point)
        graph_this.add_edge(grasp_point_id, pivot_point_id, POS_NONE)

        graph_others = self.cg.create_compound_graph_except(cableID)
        num_other_cables = len(self.cg.graphs) - 1
        save_path = f"cableGraphs/numcx_composite_transitioned.png"
        path = None
        if vis is True:
            composite_graph = graph_this.compose(graph_others)
            if save is True:
                path = composite_graph.visualize(save_path=save_path)
            else:
                composite_graph.visualize()
        num_cx = graph_this.get_num_crossings_two_graphs(graph_others)
        # also subtract repeated count at the fixed endpoint
        return num_cx - num_other_cables, save_path

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
        save_path = None
        use_min = False
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
                num_cx_orig = self.cg.compound_graph.get_num_crossings()
                num_cx_new, _ = self.get_num_crossings(
                    pivot_point_id,
                    grasp_point_id,
                    res_point_free_endpoint.tolist(),
                    cableID,
                )
                if num_cx_new < 0:
                    use_min = True
                tmp_elim_num = num_cx_orig - num_cx_new
                if tmp_elim_num > 0:
                    res_ok = True
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
                )
                _, _ = self.get_num_crossings(
                    pivot_point_id,
                    grasp_point_id,
                    res_point.tolist(),
                    cableID,
                    vis=False,
                    save=False,
                )
            # for plotting
            if i == 0:
                res_ok = False
                theta_range_tmp = []
                elim_num = 0
                # plot the zero theta and save
                _, save_path = self.get_num_crossings(
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
            return None
        # only return the theta ranges with the biggest elim num
        max_elim_num = np.max(list(theta_ranges.keys()))
        min_elim_num = np.min(list(theta_ranges.keys()))
        # draw on fig the theta ranges
        angle0 = self.get_zero_theta_vector_angle(grasp_point, pivot_point)
        # for plotting
        if save_path is not None:
            # load image
            plt.figure()
            plt.imshow(mpimg.imread(save_path))
            # plt.plot(
            #     2 * pivot_point[0] + 13,
            #     2 * pivot_point[1] - 5,
            #     color="red",
            #     marker="o",
            # )
            cmap = cm.cool
            for elim, theta_range in theta_ranges.items():
                rgb = cmap(elim / 3)
                for rang in theta_range:
                    arc_angles = (
                        np.linspace(rang[0] - 0.14, rang[1] - 0.14, 50) + angle0
                    )
                    arc_xs = 2 * (
                        pivot_point[0]
                        + 16
                        + (grasp_length + 7) * np.cos(arc_angles)
                    )
                    arc_ys = 2 * (
                        pivot_point[1]
                        - 7
                        + (grasp_length + 7) * np.sin(arc_angles)
                    )
                    plt.plot(arc_xs, arc_ys, color=rgb, lw=4)
            bounds = [0, 1, 2, 3]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            plt.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                label=r"|$V_{elim}$|",
                orientation="vertical",
            )
            plt.savefig(f"cableGraphs/composite_with_action_space.png")
            plt.show()
        return (
            theta_ranges[min_elim_num]
            if use_min
            else theta_ranges[max_elim_num]
        )

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
                cost += (
                    self.weight_dist * dist_cost + self.weight_curv * curv_cost
                )
        return cost

    def search_goal_coord(self, grasp_point_id, keypoint_id, cableID):
        # imagine pulling tight the cable segment from grasp point to pivot
        graph: Graph = self.cg.graphs[cableID]

        # the pivot point is the predecessor of the keypoint (an undercx)
        pivot_point_id = graph.get_pred(keypoint_id)
        print("grasp: ", grasp_point_id, "pivot: ", pivot_point_id)

        grasp_length = graph.compute_length(grasp_point_id, pivot_point_id)
        total_length = graph.compute_length(
            graph.get_free_endpoint(), pivot_point_id
        )

        # draw a circle (or multiple arcs on a circle) around pivot point.
        # this is the action space
        theta_ranges = self.generate_action_space(
            grasp_point_id, pivot_point_id, grasp_length, total_length, cableID
        )
        print(theta_ranges)
        if theta_ranges is None:
            return None, None
        thetas = []
        costs = []
        fig, axs = plt.subplots(1, len(theta_ranges), squeeze=False)
        fig.suptitle("Cost in action subspace")
        for i, theta_range in enumerate(theta_ranges):
            res = minimize_scalar(
                self.calc_cost,
                args=(total_length, grasp_point_id, pivot_point_id, cableID),
                bounds=theta_range,
                method="bounded",
            )
            print(res)
            if res.success is True:
                thetas.append(res.x)
                costs.append(res.fun)
            thetas_ = np.linspace(theta_range[0], theta_range[1], num=60)
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
            axs[0, i].plot(thetas_, all_costs)
            axs[0, i].plot(res.x, res.fun, color="red", marker="o")
        print(thetas)
        if len(thetas) == 0:
            return None
        for ax in axs.flat:
            ax.set(xlabel=r"$\theta$", ylabel="Cost")
        plt.savefig(f"cableGraphs/cost.png")
        plt.show()

        theta = thetas[np.argmin(costs)]
        print(theta)
        # convert theta into goal coord
        goal_coord, goal_vec = self.theta_to_goal_coord(
            theta, grasp_point_id, pivot_point_id, grasp_length, cableID
        )
        return goal_coord, goal_vec

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
                        [
                            action.pick_vec[0],
                            action.pick_vec[1],
                            0,
                        ]
                    )
                    action.place_vec_3d = np.array(
                        [
                            action.place_vec[0],
                            action.place_vec[1],
                            0,
                        ]
                    )
                    self.fa.exe_action(action)
                    return UNWEAVE_IN_PROGRESS
        if num_done == len(self.cg.graphs.keys()):
            print("Unweaving all done. Stopping...")
            return UNWEAVE_ALL_DONE
        return UNWEAVE_FAIL

    def run(self):
        if self.use_rs is False:
            print("Realsense not in use, returning")
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
            self.workspace = [
                (0, 0),
                (self.width, self.height),
            ]
            cv2.imshow("img", bgr)
            cv2.waitKey(0)
            cv2.imwrite("cableImages/rs_cable_imgs2/test.png", bgr)
            res = self.unweave_step(bgr, depth)
            if res is UNWEAVE_FAIL:
                raise RuntimeError("Cannot find an action to unweave!")
            elif res is UNWEAVE_ALL_DONE:
                break
            self.cg = CableGraph()
        print("Done unweaving all cables")
        self.realsense.close()


# test
def test_simple_policy():
    img = cv2.imread("d:/XinyuWang/2022_Fall/16740/cable_manipulation/cableImages/generated_02.png")
    img_w = img.shape[1]
    img_h = img.shape[0]
    csp = CableSimplePolicy(width=img_w, height=img_h, use_rs=False)
    csp.gen_graph_from_image(img)
    action = csp.eliminate_crossing("cable_yellow")
    print(action)


if __name__ == "__main__":
    # csp = CableSimplePolicy(use_rs=True)
    # csp.run()
    test_simple_policy()
