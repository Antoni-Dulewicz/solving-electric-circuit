import networkx as nx
import random
import networkx as nx
import random

#generowanie grafu losowego spojnego
def generating_random_graph(n,weight_max,p=0.5):
    G = nx.gnp_random_graph(n,p)
    
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        
        for i in range(1, len(components)):
            node1 = random.choice(list(components[i - 1]))
            node2 = random.choice(list(components[i]))
            weight = random.randint(0.1, weight_max)
            G.add_edge(node1, node2, weight=weight)
    
    for u, v, d in G.edges(data=True):
        if 'weight' not in d:
            d['weight'] = random.randint(1, weight_max)
    
    edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    return edges



#generowanie siatki 2D
def generating_random_grid_graph(rows,cols,max_weight):
    graph = nx.grid_2d_graph(rows,cols)

    for (u,v) in graph.edges():
        graph[u][v]['weight'] = random.uniform(0.1, max_weight)

    edges = [(u[0]*cols+u[1], v[0]*cols+v[1], graph[u][v]['weight']) for u, v in graph.edges()]

    return edges

#generowanie grafu losowego regularnego
def generating_random_regular_graph(n, max_weight):
    graph = nx.DiGraph()
    for i in range(n):
        graph.add_node(i)

    for i in range(n):
        for j in range(i+1, n):
            curr_weight = random.uniform(0.1, max_weight)
            tmp = random.uniform(0,1)
            if tmp > 0.5:
                graph.add_edge(i, j, weight=curr_weight)
            else:
                graph.add_edge(j, i, weight=curr_weight)

    edges = [(u, v, graph[u][v]['weight']) for u, v in graph.edges()]
    return edges

#graf z mostkiem
def generating_random_bridge_graph(n,min_weight, max_weight):
    graph1 = nx.DiGraph()
    graph2 = nx.DiGraph()

    m = n//2

    for i in range(m):
        graph1.add_node(i)
        graph2.add_node(i+ m)

    for i in range(m):
        for j in range(i+1, m):
            curr_weight = random.uniform(min_weight, max_weight)
            tmp = random.uniform(0,1)
            if tmp > 0.5:
                graph1.add_edge(i, j, weight=curr_weight)
                graph2.add_edge(j+m,i+m , weight=curr_weight)
            else:
                graph1.add_edge(j, i, weight=curr_weight)
                graph2.add_edge(i+m,j+m , weight=curr_weight)

    
    weight = random.uniform(min_weight, max_weight)
    graph1.add_edge(random.choice(list(graph1.nodes())),random.choice(list(graph2.nodes())) , weight=weight)
    

    
    merged_graph = nx.compose(graph1, graph2)

    edges = [(u, v, merged_graph[u][v]['weight']) for u, v in merged_graph.edges()]
    return edges
