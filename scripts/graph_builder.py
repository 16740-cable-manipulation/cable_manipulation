import networkx as nx
import json
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utility import (
    calcDistance,
    is_line_segments_intersect,
    angle_between,
    unit_vector,
)

POS_UP = 0
POS_DOWN = 1
POS_NONE = 2

NODE_CROSSING = 3
NODE_FREE = 4
NODE_ENDPOINT = 5

id_counter = -1

DPI = 80


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

    def __init__(
        self,
        width=640,
        height=480,
        free_endpoint=[],
        fixed_endpoint=[],
        cables=[],
    ):
        self.G = nx.DiGraph(
            free_endpoint=copy.deepcopy(free_endpoint),
            fixed_endpoint=copy.deepcopy(fixed_endpoint),
            cables=copy.deepcopy(cables),
        )
        self.width = width
        self.height = height

    def save(self, save_path):
        """Save the graph to disk."""
        data = nx.node_link_data(self.G)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, save_path):
        """Load the graph from a file."""
        with open(save_path, "r") as f:
            self.G = nx.node_link_graph(json.load(f))

    def add_edge(self, id1, id2, pos, color="black"):
        """Add an edge from id1 to id2"""
        assert self.has_node(id1) and self.has_node(id2)
        self.G.add_edge(id1, id2, pos=pos, color=color)

    def add_node(self, type, coords=[]):
        id = self._generate_next_available_id()
        self.G.add_node(id, type=type, coords=copy.deepcopy(coords))
        return id

    def copy_node(self, other: "Graph", id):
        """Copy a node from this graph to another graph"""
        assert self.has_node(id)
        node_data = copy.deepcopy(self.G.nodes[id])
        other.G.add_node(id, **node_data)

    def copy_edge(self, other: "Graph", id1, id2):
        """Copy an edge from this graph to another graph"""
        assert self.has_edge(id1, id2)
        edge_data = copy.deepcopy(self.G.edges[id1, id2])
        other.G.add_edge(id1, id2, **edge_data)

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

    def get_fixed_endpoint(self):
        assert len(self.G.graph["fixed_endpoint"]) > 0
        return self.G.graph["fixed_endpoint"][0]

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
        return [self.get_succ(id, pos=pos), self.get_pred(id, pos=pos)]

    def get_succ(self, id, pos=POS_NONE):
        """Get a single successor. If id is a crossing, pos will
        be the selection criterion"""
        assert self.has_node(id)
        ls = list(self.G.successors(id))
        if pos == POS_NONE:
            return ls[0]
        else:
            return list(
                filter(
                    lambda succ: self.get_pos_label(id, succ) == pos
                    or self.get_pos_label(id, succ) == POS_NONE,
                    ls,
                )
            )[0]

    def get_pred(self, id, pos=POS_NONE):
        """Get a single predecessor. If id is a crossing, pos will
        be the selection criterion"""
        assert self.has_node(id)
        ls = list(self.G.predecessors(id))
        if pos == POS_NONE:
            return ls[0]
        else:
            return list(
                filter(
                    lambda pred: self.get_pos_label(pred, id) == pos
                    or self.get_pos_label(pred, id) == POS_NONE,
                    ls,
                )
            )[0]

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
        assert self.has_node(id), f"id = {id} not in graph"
        return copy.deepcopy(self.G.nodes[id]["coords"])

    def compute_length(self, id1, id2, pos=POS_NONE):
        """Compute the length of a cable segment from id1 to id2, where id1 is
        before id2.

        If id1 is a crossing, pos will select the next direction
        """
        assert self.has_node(id1) and self.has_node(id2)
        pred = id1
        length = 0
        while pred != id2:
            succ = self.get_succ(pred, pos=pos)
            pred_coord = self.get_node_coords(pred)
            succ_coord = self.get_node_coords(succ)
            length += calcDistance(
                pred_coord[0], pred_coord[1], succ_coord[0], succ_coord[1]
            )
            pred = succ
        return length

    def calc_tangent_vec(self, id):
        assert self.has_node(id)
        pred = np.array(self.get_node_coords(self.get_pred(id)))
        succ = np.array(self.get_node_coords(self.get_succ(id)))
        this_coord = np.array(self.get_node_coords(id))
        return (
            (unit_vector(succ - this_coord) + unit_vector(this_coord - pred))
            / 2
        ).tolist()

    def grow_branch(self, coords, id, div=1):
        """Grow a branch from coords to id, where id is already in the graph.
        The newly added nodes have float coordinates

        ``div``: number of edges in this newly grown branch
        """
        assert self.has_node(id) and div > 0
        last_coord = self.get_node_coords(id)
        edge_length = (
            calcDistance(last_coord[0], last_coord[1], coords[0], coords[1])
            / div
        )
        vec = unit_vector(np.array(last_coord) - np.array(coords)) * edge_length
        this_coord = np.array(coords)
        this_id = self.add_node(NODE_FREE, coords=this_coord.tolist())
        for i in range(div):
            if i < div - 1:
                next_coord = this_coord + vec
                next_id = self.add_node(NODE_FREE, coords=next_coord.tolist())
                self.add_edge(this_id, next_id, pos=POS_NONE)
                this_id = next_id
                this_coord = next_coord
            else:  # for the last edge, just connect to the id
                self.add_edge(this_id, id, pos=POS_NONE)

    def get_next_keypoint(self, id, pos=POS_NONE):
        """Get the id of the next crossing or endpoint and the crossing pos.

        Return the next keypoint and a list of free nodes starting from id,
        including id if id is free.

        If the id is a crossing, ``pos`` argument has to be provided
        TODO this is buggy when pos isn't pos_none
        """
        if self.is_fixed_endpoint(id):
            return None, POS_NONE, []
        free_nodes = []
        if self.is_free(id):
            free_nodes.append(id)
        # elif self.is_crossing(id):
        #     if pos == POS_NONE:
        #         raise RuntimeError(
        #             "Need pos lable if the starting node is a crossing"
        #         )
        while True:
            pred = id
            id = self.get_succ(id, pos=pos)
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
        TODO this is buggy when pos isn't pos_none
        """
        next_id, cx_pos, nodes = self.get_next_keypoint(id, pos=pos)
        if (
            next_id is None
            or cx_pos == POS_DOWN
            or self.is_fixed_endpoint(next_id)
        ):
            return next_id, nodes
        while True:
            # print(next_id, cx_pos)
            # TODO handle self occlusion
            next_id, cx_pos, new_nodes = self.get_next_keypoint(next_id)
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
        if (
            len(self.G.graph["fixed_endpoint"]) == 0
            or len(other.G.graph["fixed_endpoint"]) == 0
            or self.G.graph["fixed_endpoint"][0]
            != other.G.graph["fixed_endpoint"][0]
        ):
            res.G.graph["fixed_endpoint"] = (
                self.G.graph["fixed_endpoint"] + other.G.graph["fixed_endpoint"]
            )
        res.G.graph["cables"] = self.G.graph["cables"] + other.G.graph["cables"]
        res.width = other.width
        res.height = other.height
        return res

    def build_subgraph(self, id1, id2, pos=POS_NONE):
        """Build a subgraph from id1 to id2, where id1 is
        before id2.

        If id1 is a crossing, pos will select the next direction
        """
        assert self.has_node(id1) and self.has_node(id2)
        graph = Graph()
        pred = id1
        self.copy_node(graph, pred)
        while pred != id2:
            succ = self.get_succ(pred, pos=pos)
            self.copy_node(graph, succ)
            self.copy_edge(graph, pred, succ)
            pred = succ
        return graph

    def find_nearest_node(self, coord):
        best_dist = None
        best_id = None
        for id in self.get_nodes():
            node_coord = self.get_node_coords(id)
            dist = calcDistance(
                node_coord[0], node_coord[1], coord[0], coord[1]
            )
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = id
        return best_dist, best_id

    def calc_distance_between_graphs(self, other: "Graph"):
        """For every node in a graph, find the closest node in the other
        graph and compute the distance, (then average them)?"""
        sum_dist = 0
        for id_this in self.get_nodes():
            dist, id_that = other.find_nearest_node(
                self.get_node_coords(id_this)
            )
            sum_dist += dist
        return sum_dist / len(self.get_nodes())

    def calc_curvature(self, id, pos=POS_NONE):
        """Currently this is calculating the angle at id, not curvature"""
        assert self.has_node(id) and not self.is_endpoint(id)
        node = np.array(self.get_node_coords(id))
        succ = np.array(self.get_node_coords(self.get_succ(id, pos=pos)))
        pred = np.array(self.get_node_coords(self.get_pred(id, pos=pos)))
        return angle_between(succ - node, pred - node)

    def simplify(self):
        pass

    def get_num_crossings(self):
        """Get the number of crossings within this graph"""
        num = 0
        for id in self.get_nodes():
            if self.is_crossing(id):
                num += 1
        return num

    def is_edge_intersect(self, edge1, edge2):
        edge1_pt1 = self.get_node_coords(edge1[0])
        edge1_pt2 = self.get_node_coords(edge1[1])
        edge2_pt1 = self.get_node_coords(edge2[0])
        edge2_pt2 = self.get_node_coords(edge2[1])
        return is_line_segments_intersect(
            edge1_pt1, edge1_pt2, edge2_pt1, edge2_pt2
        )

    def get_num_crossings_two_graphs(self, other: "Graph"):
        """Get #cx by checking edge overlap regardless of node type
        TODO this is not robust to edge cases"""
        num = 0
        this_edges = self.get_edges()
        others_edges = other.get_edges()
        composite_graph = self.compose(other)
        num_cx_others = 0
        cx_others = composite_graph.get_crossings()
        for cx in cx_others:
            if len(list(composite_graph.G.successors(cx))) >= 2:
                num_cx_others += 1

        for edge_this in this_edges:
            for edge_that in others_edges:
                if composite_graph.is_edge_intersect(edge_this, edge_that):
                    num += 1
        # print(num, num_cx_others)
        return num - num_cx_others * 3

    def get_num_crossings_with_line_seg(self, pt1, pt2):
        """Assuming the line seg never intersects an edge exactly on the node"""
        num = 0
        for edge in self.get_edges():
            edge_pt1 = self.get_node_coords(edge[0])
            edge_pt2 = self.get_node_coords(edge[1])
            if is_line_segments_intersect(edge_pt1, edge_pt2, pt1, pt2):
                num += 1
        return num

    def visualize(self, ax=None, save_path=None):
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
            550
            if self.is_crossing(id)
            else 400
            if self.is_endpoint(id)
            else 300
            for id in self.get_nodes()
        ]
        widths = [
            3.0
            if self.get_pos_label(edge[0], edge[1]) == POS_DOWN
            else 5.0
            if self.get_pos_label(edge[0], edge[1]) == POS_UP
            else 2.0
            for edge in self.get_edges()
        ]

        pos = {}
        for id in self.get_nodes():
            tmp_pos = self.get_node_coords(id)
            # flip y coordinate
            tmp_pos[1] = self.height - 1 - tmp_pos[1]
            pos[id] = tmp_pos
        if ax is None:
            fig = plt.figure(
                1,
                figsize=(self.width / DPI, self.height / DPI),
                dpi=DPI,
            )
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
            if save_path is not None:
                plt.savefig(save_path)
            plt.show()
            return None
        else:
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
                ax=ax,
            )
            # plt.show()
            # return ax


class CableGraph:
    def __init__(self):
        # a collection of graphs, one for each cable
        self.graphs = {}
        self.compound_graph = None

    def create_compound_graph(self):
        for _, graph in self.graphs.items():
            if self.compound_graph is not None:
                self.compound_graph = self.compound_graph.compose(graph)
            else:
                self.compound_graph = copy.deepcopy(graph)

    def create_compound_graph_except(self, cableID):
        assert cableID in self.graphs.keys()
        compound_graph = None
        for cableid, graph in self.graphs.items():
            if cableid != cableID:
                if compound_graph is not None:
                    compound_graph = compound_graph.compose(graph)
                else:
                    compound_graph = copy.deepcopy(graph)
        return compound_graph  # a graph or none

    def create_graphs(self, cables_data):
        """Create a graph for each cable, where the crossings and the fixed
        endpoint vertices have shared node ID across graphs

        Input:
        ``cables_data``: dict of the form {cableID1: data1, cableID2: data2, },
        where data is a dict of the form
        {"coords": coords, "pos": pos, "cx": cx, "color": color,
        "width": width, "height": height}

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

        width, heigh: image size
        """
        cx_coord_id_map = {}
        # fixed_endpoint_id = None
        for cableID, data in cables_data.items():
            coords = data["coords"]
            pos = data["pos"]
            cx = data["cx"]
            color = data["color"]
            w = data["width"]
            h = data["height"]
            graph, fixed_endpoint_id = self.create_graph(
                cableID,
                coords,
                pos,
                cx,
                cx_coord_id_map,
                color=color,
                width=w,
                height=h,
                # fixed_endpoint_id=fixed_endpoint_id,
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
        width=640,
        height=480,
        fixed_endpoint_id=None,
    ):
        assert len(coords) >= 3
        graph = Graph(cables=[cableID], width=width, height=height)
        pred_id = None
        for i, coord in enumerate(coords):
            if i == 0 or i == len(coords) - 1:
                if i == 0:
                    id = graph.add_node(NODE_ENDPOINT, coord)
                    graph.add_free_endpoint(id)
                else:
                    if fixed_endpoint_id is None:
                        id = graph.add_node(NODE_ENDPOINT, coord)
                        fixed_endpoint_id = id
                    else:
                        id = fixed_endpoint_id
                        graph.add_node_id(id, NODE_ENDPOINT, coord)
                    graph.add_edge(pred_id, id, pos[i - 1])
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
        return graph, fixed_endpoint_id


def test_graph():
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
    data1 = {
        "coords": coords1,
        "pos": pos1,
        "cx": cx1,
        "color": "blue",
        "width": 20,
        "height": 20,
    }
    # cables_data = {"cable1": data1}
    # cg.create_graphs(cables_data)
    # cg.graphs["cable1"].visualize()

    coords2 = np.array(
        [[0, 6], [0, 5], [1, 4], [2, 3.5], [3, 4], [4, 5], [5, 5]]
    )
    cx2 = [[3, 4], [1, 4]]
    coords2 = coords2.tolist()
    pos2 = [POS_NONE] * 7
    pos2[2] = POS_DOWN
    pos2[4] = POS_UP
    data2 = {
        "coords": coords2,
        "pos": pos2,
        "cx": cx2,
        "color": "red",
        "width": 20,
        "height": 20,
    }

    cables_data = {"cable1": data1, "cable2": data2}
    cg.create_graphs(cables_data)
    cg.create_compound_graph()
    print(cg.compound_graph.get_neighbors(2, pos=POS_DOWN))
    print(cg.graphs["cable1"].get_succ(6, pos=POS_UP))
    print(cg.graphs["cable2"].get_next_fixed_keypoint(12))
    print(cg.compound_graph.get_crossings())
    cg.compound_graph.visualize()


def reset_id():
    global id_counter
    id_counter = -1


if __name__ == "__main__":
    cg = CableGraph()
    from cable_discretization import getCablesDataFromImage

    img = cv2.imread("cableImages/generated_03.png")
    cables_data = getCablesDataFromImage(img, vis=False)
    print(cables_data)
    cg.create_graphs(cables_data)
    # cg.graphs["cable_red"].visualize(save_path="cableGraphs/red.png")
    # cg.graphs["cable_blue"].visualize(save_path="cableGraphs/blue.png")
    # cg.graphs["yellow"].visualize()
    cg.create_compound_graph()
    cg.compound_graph.visualize()
    print(
        cg.graphs["cable_red"].get_num_crossings_two_graphs(
            cg.graphs["cable_blue"]
        )
    )
    # g2 = cg.create_compound_graph_except("cable_red")
    # g2.visualize()

    # print(cg.graphs["cable_red"].G.graph["free_endpoint"])
    # print(cg.graphs["cable_red"].G.graph["fixed_endpoint"])
    # print(cg.graphs["cable_blue"].G.graph["free_endpoint"])
    # print(cg.graphs["cable_blue"].G.graph["fixed_endpoint"])
    # endid = cg.graphs["cable_blue"].get_fixed_endpoint()
    # subg = cg.graphs["cable_blue"].build_subgraph(27, endid)
    # subg.visualize()
    # dist = cg.graphs["cable_red"].calc_distance_between_graphs(
    #     cg.graphs["cable_blue"]
    # )
    # print(dist)
