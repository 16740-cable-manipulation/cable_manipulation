import networkx as nx
import json
import copy
import matplotlib.pyplot as plt
import numpy as np

POS_UP = 0
POS_DOWN = 1
POS_NONE = 2

NODE_CROSSING = 3
NODE_FREE = 4
NODE_ENDPOINT = 5

id_counter = -1


class Graph:
    """Graph of a single cable

    A graph containing N nodes (vertex, v) and at least N-1 edges (e).
    Each node is assigned a globally unique ID.
    The graph properties include its cable ID and its endpoint nodes.

    Each vertex stores the coordinates of a point.
    Each edges stores a position label.

    If a vertex v is an overcrossing, both of its edges are POS_UP.
    Similarly, if v is an undercrossing, both of its edges are POS_DOWN.
    If an edge is not connected to a crossing or an endpoint, it is POS_NONE.

    The graph can be simplified from a crude graph into a minimal graph.

    If a vertex is not a crossing or an endpoint and.. (distance condition?),
    then it is graspable.
    """

    def __init__(self, free_endpoint=[], fixed_endpoint=[], cables=[]):
        self.G = nx.DiGraph(
            free_endpoint=copy.deepcopy(free_endpoint),
            fixed_endpoint=copy.deepcopy(fixed_endpoint),
            cables=copy.deepcopy(cables),
        )

    def save(self, save_path):
        """Save the graph to disk."""
        data = nx.node_link_data(self.G)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, save_path):
        """Load the graph from a file."""
        with open(save_path, "r") as f:
            self.G = nx.node_link_graph(json.load(f))

    def add_edge(self, id1, id2, pos):
        """Add an edge from id1 to id2"""
        assert self.has_node(id1) and self.has_node(id2)
        self.G.add_edge(id1, id2, pos=pos)

    def add_node(self, type, coords=[]):
        id = self._generate_next_available_id()
        self.G.add_node(id, type=type, coords=copy.deepcopy(coords))
        return id

    def add_node_id(self, id, type, coords=[]):
        self.G.add_node(id, type=type, coords=copy.deepcopy(coords))

    def has_node(self, id):
        return id in self.G

    def has_edge(self, id1, id2):
        """Whether there is an edge from id1 to id2"""
        return (
            self.has_node(id1) and self.has_node(id2) and id2 in self.G.adj[id1]
        )

    def _generate_next_available_id(self):
        """Generate a globally unique ID"""
        # id = shortuuid.uuid()[:5]
        global id_counter
        id_counter += 1
        return id_counter

    def is_crossing(self, id):
        return self.has_node(id) and self.G.nodes[id]["type"] == NODE_CROSSING

    def is_endpoint(self, id):
        return self.has_node(id) and self.G.nodes[id]["type"] == NODE_ENDPOINT

    def is_free(self, id):
        return (
            self.has_node(id)
            and not self.is_crossing(id)
            and not self.is_endpoint(id)
        )

    def is_fixed_endpoint(self, id):
        return id in self.G.graph["fixed_endpoint"]

    def get_free_endpoint(self):
        assert len(self.G.graph["free_endpoint"]) > 0
        return self.G.graph["free_endpoint"][0]

    def get_pos_label(self, id1, id2):
        assert self.has_edge(id1, id2)
        return self.G.edges[id1, id2]["pos"]

    def get_neighbors(self, id, pos=POS_NONE):
        """Get both the predecessor and the successor

        ``pos``: Specify the position label of the edges between this node
        and its neighbors. If pos is POS_NONE, all preds & successors will be
        returned.
        """
        assert self.has_node(id)
        return self.get_succ(id, pos=pos) + self.get_pred(id, pos=pos)

    def get_succ(self, id, pos=POS_NONE):
        """Get a list of successor(s). If id is a crossing, pos will
        be the selection criterion"""
        assert self.has_node(id)
        ls = list(self.G.successors(id))
        if pos == POS_NONE:
            return ls
        else:
            return list(
                filter(
                    lambda succ: self.get_pos_label(id, succ) == pos
                    or self.get_pos_label(id, succ) == POS_NONE,
                    ls,
                )
            )

    def get_pred(self, id, pos=POS_NONE):
        """Get a list of predecessor(s). If id is a crossing, pos will
        be the selection criterion"""
        assert self.has_node(id)
        ls = list(self.G.predecessors(id))
        if pos == POS_NONE:
            return ls
        else:
            return list(
                filter(
                    lambda pred: self.get_pos_label(pred, id) == pos
                    or self.get_pos_label(pred, id) == POS_NONE,
                    ls,
                )
            )

    def get_nodes(self):
        return list(self.G.nodes)

    def get_edges(self):
        return list(self.G.edges)

    def get_crossings(self):
        return list(
            filter(lambda node: self.is_crossing(node), self.get_nodes())
        )

    def set_all_edge_color(self, color):
        dic = {}
        for edge in self.get_edges():
            dic[edge] = color
        nx.set_edge_attributes(self.G, dic, name="color")

    def get_edge_color(self, id1, id2):
        assert self.has_edge(id1, id2)
        return self.G.edges[id1, id2]["color"]

    def get_node_coords(self, id):
        """Return a deep copy of the coords list"""
        assert self.has_node(id)
        return copy.deepcopy(self.G.nodes[id]["coords"])

    def get_next_keypoint(self, id, pos=POS_NONE):
        """Get the id of the next crossing or endpoint and the crossing pos. 

        Return the next keypoint and a list of free nodes starting from id, 
        including id if id is free.
        
        If the id is a crossing, ``pos`` argument has to be provided
        """
        if self.is_fixed_endpoint(id):
            return None, POS_NONE, []
        free_nodes = []
        if self.is_free(id):
            free_nodes.append(id)
        elif self.is_crossing(id):
            if pos == POS_NONE:
                raise RuntimeError(
                    "Need pos lable if the starting node is a crossing"
                )
        while True:
            pred = id
            id = self.get_succ(id, pos=pos)[0]
            if not self.is_free(id):
                cx_pos = self.get_pos_label(pred, id)
                return id, cx_pos, free_nodes
            else:
                free_nodes.append(id)

    def get_next_fixed_keypoint(self, id, pos=POS_NONE):
        """Get the id of the next undercrossing or fixed endpoint. 

        Return the next fixed keypoint and a list of free nodes starting from 
        id, including id if id is free.
        
        If the id is a crossing, ``pos`` argument has to be provided
        """
        next_id, cx_pos, nodes = self.get_next_keypoint(id, pos=pos)
        if (
            next_id is None
            or cx_pos == POS_DOWN
            or self.is_fixed_endpoint(next_id)
        ):
            return next_id, nodes
        while True:
            next_id, cx_pos, new_nodes = self.get_next_keypoint(
                next_id, pos=cx_pos
            )
            nodes.extend(new_nodes)
            if (
                next_id is None
                or cx_pos == POS_DOWN
                or self.is_fixed_endpoint(next_id)
            ):
                return next_id, nodes

    def add_free_endpoint(self, id):
        assert self.has_node(id)
        self.G.graph["free_endpoint"].append(id)

    def add_fixed_endpoint(self, id):
        assert self.has_node(id)
        self.G.graph["fixed_endpoint"].append(id)

    def compose(self, other: "Graph"):
        res = Graph()
        res.G = nx.compose(self.G, other.G)
        res.G.graph["free_endpoint"] = (
            self.G.graph["free_endpoint"] + other.G.graph["free_endpoint"]
        )
        res.G.graph["fixed_endpoint"] = (
            self.G.graph["fixed_endpoint"] + other.G.graph["fixed_endpoint"]
        )
        res.G.graph["cables"] = self.G.graph["cables"] + other.G.graph["cables"]
        return res

    def simplify(self):
        pass

    def visualize(self, save_path=None):
        colors = [
            "orange"
            if self.is_crossing(id)
            else "purple"
            if self.is_endpoint(id)
            else "gray"
            for id in self.get_nodes()
        ]
        e_colors = [
            self.get_edge_color(edge[0], edge[1]) for edge in self.get_edges()
        ]
        sizes = [
            1000
            if self.is_crossing(id)
            else 700
            if self.is_endpoint(id)
            else 500
            for id in self.get_nodes()
        ]
        widths = [
            3.0
            if self.get_pos_label(edge[0], edge[1]) == POS_DOWN
            else 6.0
            if self.get_pos_label(edge[0], edge[1]) == POS_UP
            else 1.0
            for edge in self.get_edges()
        ]

        pos = {}
        for id in self.get_nodes():
            pos[id] = self.get_node_coords(id)

        nx.draw(
            self.G,
            pos=pos,
            node_color=colors,
            edge_color=e_colors,
            width=widths,
            with_labels=True,
            font_color="white",
            font_weight="bold",
            node_size=sizes,
        )
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)


class CableGraph:
    def __init__(self):
        # a collection of graphs, one for each cable
        self.graphs = {}
        self.compound_graph = Graph()

    def create_compound_graph(self):
        graph_prev = None
        for _, graph in self.graphs.items():
            if graph_prev is not None:
                self.compound_graph = graph.compose(graph_prev)
            graph_prev = graph

    def create_graphs(self, cables_data):
        """Create a graph for each cable, where the crossings and the fixed
        endpoint vertices have shared node ID across graphs

        Input:
        ``cables_data``: dict of the form {cableID1: data1, cableID2: data2, },
        where data is a dict of the form
        {"coords": coords, "pos": pos, "cx": cx, "color": color}

        coords: a N*2 2D list (not np array) of all N inter-connected
        discretization points along the cable, whose first coord is the
        free endpoint and the last coord is the fixed endpoint

        pos: a 1D list of length N which gets values from
        {POS_UP, POS_DOWN, POS_NONE}, depending on whether the coord is
        an overcrossing, an undercrossing, or neither.

        cx: a M*2 2D list of the coordinates of the M crossings.
        (Note: "coords" might contain repetitive coordinates, but those should
        be listed in cx.)

        color: a string representing the cable color.
        """
        cx_coord_id_map = {}
        for cableID, data in cables_data.items():
            coords = data["coords"]
            pos = data["pos"]
            cx = data["cx"]
            color = data["color"]
            graph = self.create_graph(
                cableID, coords, pos, cx, cx_coord_id_map, color=color
            )
            self.graphs[cableID] = graph

    def create_graph(
        self,
        cableID,
        coords,
        pos,
        cx,
        cx_coord_id_map,
        color="black",
        fixed_endpoint_id=None,
    ):
        assert len(coords) >= 3
        graph = Graph(cables=[cableID])
        pred_id = None
        for i, coord in enumerate(coords):
            if i == 0 or i == len(coords) - 1:
                id = graph.add_node(NODE_ENDPOINT, coord)
                if i == 0:
                    graph.add_free_endpoint(id)
                else:
                    graph.add_edge(pred_id, id, pos[i - 1])
                    if fixed_endpoint_id is None:
                        fixed_endpoint_id = id
                    graph.add_fixed_endpoint(fixed_endpoint_id)
            else:
                if coord in cx:
                    id = cx_coord_id_map.get(tuple(coord))
                    if id is None:
                        id = graph.add_node(NODE_CROSSING, coord)
                        cx_coord_id_map[tuple(coord)] = id
                    else:
                        graph.add_node_id(id, NODE_CROSSING, coord)
                    graph.add_edge(pred_id, id, pos[i])
                else:
                    id = graph.add_node(NODE_FREE, coord)
                    graph.add_edge(pred_id, id, pos[i - 1])
            pred_id = id
        graph.set_all_edge_color(color)
        return graph


# test building a graph for two cables
cg = CableGraph()
coords1 = np.array(
    [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [3, 4],
        [2, 5],
        [1, 4],
        [1, 3],
        [2, 2],
        [3, 2],
        [4, 2],
    ]
)
cx1 = [[2, 2], [3, 4], [1, 4]]
coords1 = coords1.tolist()
pos1 = [POS_NONE] * 11
pos1[2] = POS_DOWN
pos1[4] = POS_DOWN
pos1[6] = POS_UP
pos1[8] = POS_UP
data1 = {"coords": coords1, "pos": pos1, "cx": cx1, "color": "blue"}
# cables_data = {"cable1": data1}
# cg.create_graphs(cables_data)
# cg.graphs["cable1"].visualize()

coords2 = np.array([[0, 6], [0, 5], [1, 4], [2, 3.5], [3, 4], [4, 5], [5, 5]])
cx2 = [[3, 4], [1, 4]]
coords2 = coords2.tolist()
pos2 = [POS_NONE] * 7
pos2[2] = POS_DOWN
pos2[4] = POS_UP
data2 = {"coords": coords2, "pos": pos2, "cx": cx2, "color": "red"}


cables_data = {"cable1": data1, "cable2": data2}
cg.create_graphs(cables_data)
cg.create_compound_graph()
print(cg.compound_graph.get_neighbors(2, pos=POS_DOWN))
print(cg.graphs["cable1"].get_succ(6, pos=POS_UP))
print(cg.graphs["cable2"].get_next_fixed_keypoint(12))
print(cg.compound_graph.get_crossings())
cg.compound_graph.visualize()
