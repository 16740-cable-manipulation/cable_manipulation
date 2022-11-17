import networkx as nx
import json
import copy
import shortuuid
import matplotlib.pyplot as plt

POS_UP = 0
POS_DOWN = 1
POS_NONE = 2

NODE_CROSSING = 3
NODE_FREE = 4
NODE_ENDPOINT = 5


class Graph:
    """Graph of a single cable

    A linear graph containing N nodes (vertex, v) and N-1 edges (e).
    Each node is assigned a globally unique ID.
    Each vertex has at most 2 edges.
    The graph properties include its cable ID and its endpoint node ID.

    Each vertex stores the coordinates of a point.
    Each edges stores a position label.

    If a vertex v is an overcrossing, both of its edges are POS_UP.
    Similarly, if v is an undercrossing, both of its edges are POS_DOWN.
    If an edge is not connected to a crossing or an endpoint, it is POS_NONE.

    The graph can be simplified from a crude graph into a minimal graph.

    If a vertex is not a crossing or an endpoint and.. (distance condition?),
    then it is graspable.
    """

    def __init__(self):
        self.G = nx.Graph()

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
        assert self.has_node(id1) and self.has_node(id2)
        self.G.add_edge(id1, id2, pos=pos)

    def add_node(self, type, coords=[]):
        id = self._generate_next_available_id()
        self.G.add_node(id, type=type, coords=copy.deepcopy(coords))
        return id

    def add_node(self, id, type, coords=[]):
        assert self.has_node(id) is False
        self.G.add_node(id, type=type, coords=copy.deepcopy(coords))
        return id

    def has_node(self, id):
        return id in self.G

    def has_edge(self, id1, id2):
        return (
            self.has_node(id1) and self.has_node(id2) and id1 in self.G.adj[id2]
        )

    def _generate_next_available_id(self):
        """Generate a globally unique ID"""
        id = shortuuid.uuid()[:5]
        return id

    def is_crossing(self, id):
        return self.has_node(id) and self.G.nodes[id]["type"] == NODE_CROSSING

    def is_endpoint(self, id):
        return self.has_node(id) and self.G.nodes[id]["type"] == NODE_ENDPOINT

    def get_pos_label(self, id1, id2):
        assert self.has_edge(id1, id2)
        return self.G.edges[id1, id2]["pos"]

    def get_neighbors(self, id):
        assert self.has_node(id)
        return list(self.G.neighbors(id))

    def get_nodes(self):
        return list(self.G.nodes)

    def get_edges(self):
        return list(self.G.edges)

    def get_node_coords(self, id):
        """Return a deep copy of the coords list"""
        assert self.has_node(id)
        return copy.deepcopy(self.G.nodes[id]["coords"])

    def simplify(self):
        pass

    def visualize(self, save_path=None):
        colors = [
            "red"
            if self.is_crossing(id)
            else "blue"
            if self.is_endpoint(id)
            else "green"
            for id in self.get_nodes()
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
        """
        pos = {}
        for i, id in enumerate(self.get_nodes()):
            pos[id] = [i * 1.5, 0]
        """
        nx.draw(
            self.G,
            # pos=pos,
            node_color=colors,
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


graph = Graph()
graph.add_node(1, NODE_ENDPOINT, [0, 0, 1])
graph.add_node(2, NODE_FREE, [0, 0, 2])
graph.add_edge(1, 2, POS_NONE)
graph.add_node(3, NODE_CROSSING, [0, 0, 3])
graph.add_edge(2, 3, POS_DOWN)
graph.add_node(4, NODE_FREE, [0, 0, 4])
graph.add_edge(3, 4, POS_DOWN)
graph.add_node(5, NODE_ENDPOINT, [0, 0, 5])
graph.add_edge(4, 5, POS_NONE)

graph.visualize()

