from graph_builder import Graph, CableGraph
from cable_discretization import Discretize


class Action:
    def __init__(
        self,
        is_empty,
        pick_coord=[0, 0],
        pick_theta=0.0,
        place_coord=[0, 0],
        place_theta=0.0,
        z=0.0,
    ):
        self.is_empty = is_empty  # empty action does nothing
        self.pick_coord = pick_coord
        self.pick_theta = pick_theta
        self.place_coord = place_coord
        self.place_theta = place_theta
        self.z = z


class CableSimplePolicy:
    def __init__(self):
        self.cg = CableGraph()
        self.disc = Discretize()

    def gen_graph_from_image(self, img):
        cable_data = self.disc.getCablesDataFromImage(img)
        self.cg.create_graphs(cable_data)

    def eliminate_crossing(self, cableID):
        """Attempt to eliminate the first crossing of a cable.
        
        Return the action if exists, none o/w
        """
        graph: Graph = self.cg[cableID]
        next_id, nodes = graph.self.get_next_fixed_keypoint(
            self.get_free_endpoint()
        )
        if len(nodes) == 0:
            return None
        if graph.is_fixed_endpoint(next_id):
            return Action(False)
        for node in nodes:
            # attemp to move this node to free space
            goal_coord = self.search_goal_coord(node, next_id)
            if goal_coord is not None:
                return Action(True)
