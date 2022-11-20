import networkx as nx

graph = nx.DiGraph()
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)
graph.add_edges_from([(1, 2), (2, 3)])
print(list(graph.successors(2)), list(graph.predecessors(2)))
