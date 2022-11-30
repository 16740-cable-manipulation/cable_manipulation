import networkx as nx
import copy


def copy_node(one_graph, other_graph, id):
    node_data = copy.deepcopy(one_graph.nodes[id])
    other_graph.add_node(id, **node_data)


graph = nx.DiGraph()
graph.add_node(1, weight=[0.1])
graph.add_node(2, weight=[0.2])
graph.add_node(3, weight=[0.3])
graph.add_edges_from([(1, 2), (2, 3)])
print(list(graph.successors(2)), list(graph.predecessors(2)))
g2 = nx.DiGraph()
copy_node(graph, g2, 2)
print(g2.nodes[2])
g2.nodes[2]["weight"][0] = 0.4
print(g2.nodes[2])
print(graph.nodes[2])
