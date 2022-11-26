from graph_builder import Graph, CableGraph
from graph_builder import POS_DOWN, POS_UP, POS_NONE
from cable_discretization import Discretize
from action import Action
from my_franka import MyFranka
from rs_driver import Realsense


class CableSimplePolicy:
    def __init__(self, use_rs):
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

        next_id, cx_pos, nodes = graph.self.get_next_keypoint(
            self.get_free_endpoint()
        )
        if len(nodes) == 0 or cx_pos == POS_DOWN:  # first crossing is undercx
            return None
        if graph.is_fixed_endpoint(next_id):  # this cable is already done
            return Action(False)
        for node in nodes:
            # attemp to move this node to free space
            goal_coord = self.search_goal_coord(node, next_id)
            if goal_coord is not None:
                return Action(True)
        return None

    def search_goal_coord(self, grasp_point, pivot_point):
        return [0, 0]

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
