from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph_generator import generating_random_graph, generating_random_bridge_graph, generating_random_grid_graph
from enum import Enum

class Layout(Enum):
    SPRING = 1
    SPECTRAL = 2
    CIRCULAR = 3
    SPIRAL = 4
    RANDOM = 5



# reprezentacja grafu w postaci listy sasiedztwa
def switch_to_list_representation(edges):
    graph_dict = {}

    for edge in edges:
        s,t,weight = edge
        if s not in graph_dict:
            graph_dict[s] = []
        if t not in graph_dict:
            graph_dict[t] = []
        graph_dict[s].append((t,weight))
        graph_dict[t].append((s,weight))
        

    G = [[] for _ in range(len(graph_dict))]

    for s,edges in graph_dict.items():
        for edge in edges:
            G[s].append(edge)

    return G
# znajduje cykl w grafie
def find_cycle(graph, start, end):
    visited = [0 for _ in range(len(graph))]
    visited[start] = 1
    stack = [(start,[start])]

    while stack:
        u,path = stack.pop()
        
        for v,weight in graph[u]:
            if v == end:
                return path + [v]
            
            elif not visited[v]:
                    visited[v] = 1
                    stack.append((v,path + [v]))
    return None

# sprawdza czy cykl wystapil juz wczesniej
def is_duplicate(cycle,cycles):
    for c in cycles:
        if set(c) == set(cycle):
            return True
    return False

# znajduje n cykli w grafie
def find_n_cycles(edges,G,n):
    edges_cp = deepcopy(edges)
    graph = deepcopy(G)

    cycles = []

    cnt_cycles = 0

    for _ in range(n):
        #wybieramy krawedz
        for u,v,weight in edges_cp:

            #chwilowo ja usuwamy
            graph[u].remove((v,weight))
            graph[v].remove((u,weight))

            # znajdujemy sciezke pomiedzy tymi dwoma wierzcholkami
            cycle = find_cycle(graph,u,v)

            #jesli cykl istnieje i nie byl wczesniej dodany to go dodajemy
            if cycle:
                if not is_duplicate(cycle,cycles):
                    cycles.append(cycle)    
                    cnt_cycles += 1
            
            #przywracamy krawedz
            graph[u].append((v,weight))
            graph[v].append((u,weight))

            if cnt_cycles == n:
                return cycles

    return cycles

def get_edges(edges,cycle):
    new_edges = []
    for i in range(1,len(cycle)):
        u = cycle[i-1]
        v = cycle[i]
        for j in range(len(edges)):
            if (u,v) == (edges[j][0],edges[j][1]):
                new_edges.append((j,1))
                break

            if (v,u) == (edges[j][0],edges[j][1]):
                new_edges.append((j,-1))
                break
                
    u = cycle[-1]
    v = cycle[0]
    for j in range(len(edges)):
        if (u,v) == (edges[j][0],edges[j][1]):
            new_edges.append((j,1))
            break

        if (v,u) == (edges[j][0],edges[j][1]):
            new_edges.append((j,-1))
            break

    return new_edges

#znajduje krawedz do ktorej przylozone jest napiecie
def find_edge_with_voltage(edges,s,t):
    for i in range(len(edges)):
        if (edges[i][0] == s and edges[i][1] == t) or (edges[i][0] == t and edges[i][1] == s):
            return i
    return -1


def solver(edges,s,t,E):
    e = len(edges) # ile krawędzi
    G = switch_to_list_representation(edges)
    v = len(G) # ile wierzcholkow

    #krawedz do ktorej przylozone jest napiecie
    edge_with_voltage = find_edge_with_voltage(edges,s,t)


    # macierze potrzebne do rozwiazania ukladu rownan
    A = [[0.0 for _ in range(e)] for _ in range(e)]
    B = [0.0 for _ in range(e)]

    # krawedzie wychodzace i wychodzace z danych wierzcholkow - 1 prawo kir.
    for i in range(v-1):
        for j in range(e):
            if edges[j][0] == i:
                A[i][j] = -1
            elif edges[j][1] == i:
                A[i][j] = 1
    

    # rownania z oczek - 2 prawo kir.
    cycles = find_n_cycles(edges,G,e-v+1)

    i = v-1
    for cycle in cycles:
        curr_edges = get_edges(edges,cycle)
        for index,direction in curr_edges:
            if index == edge_with_voltage:
                if direction == 1:
                    B[i] -= E
                else:
                    B[i] += E
            else:
                if direction == 1:
                    A[i][index] += edges[index][2]
                else:
                    A[i][index] -= edges[index][2]
        i += 1
        

    # rozwiazujemy uklad rownan 
    A_np = np.array(A)
    B_np = np.array(B)

    return np.linalg.solve(A_np,B_np)


def edge_coloring(edges,G,currents):
    max_current = max(abs(currents))
    colors = {}
    for i in range(len(edges)):
        percentage = abs(currents[i]/max_current)
        if currents[i] > 0:
            if percentage < 0.1:
                colors[(edges[i][0],edges[i][1])] = '#B2FF66'
            elif percentage < 0.2:
                colors[(edges[i][0],edges[i][1])] = '#99FF33'
            
            elif percentage < 0.3:
                colors[(edges[i][0],edges[i][1])] = '#80FF00'

            elif percentage < 0.4:
                colors[(edges[i][0],edges[i][1])] = '#FFFF00'
            
            elif percentage < 0.5:
                colors[(edges[i][0],edges[i][1])] = '#FF8000'
                
            elif percentage < 0.6:
                colors[(edges[i][0],edges[i][1])] = '#FF0000'

            elif percentage < 0.7:
                colors[(edges[i][0],edges[i][1])] = '#CC0000'

            elif percentage < 0.8:
                colors[(edges[i][0],edges[i][1])] = '#990000'

            elif percentage < 0.9:
                colors[(edges[i][0],edges[i][1])] = '#660000'

            else:
                colors[(edges[i][0],edges[i][1])] = '#330000'
        else:
            if percentage < 0.1:
                colors[(edges[i][1],edges[i][0])] = '#B2FF66'
            elif percentage < 0.2:
                colors[(edges[i][1],edges[i][0])] = '#99FF33'
            
            elif percentage < 0.3:
                colors[(edges[i][1],edges[i][0])] = '#80FF00'

            elif percentage < 0.4:
                colors[(edges[i][1],edges[i][0])] = '#FFFF00'
            
            elif percentage < 0.5:
                colors[(edges[i][1],edges[i][0])] = '#FF8000'
                
            elif percentage < 0.6:
                colors[(edges[i][1],edges[i][0])] = '#FF0000'

            elif percentage < 0.7:
                colors[(edges[i][1],edges[i][0])] = '#CC0000'

            elif percentage < 0.8:
                colors[(edges[i][1],edges[i][0])] = '#990000'
            elif percentage < 0.9:
                colors[(edges[i][1],edges[i][0])] = '#660000'
            else:
                colors[(edges[i][1],edges[i][0])] = '#330000'

    edge_colors = [colors[edge] for edge in G.edges()]
            
    return edge_colors     

def edge_labeling(edges,currents):
    labels = {}
    for i in range(len(edges)):
        labels[(edges[i][0],edges[i][1])] = "["+str(round(currents[i],2)) + "A]" + " [" + str(round(edges[i][2],2)) + "Ohm]"
    return labels

def draw_edge_labels_with_angle(edge_labels, pos,draw_labels, ax=None, angle=0):
        if ax is None:
            ax = plt.gca()
        for edge, label in edge_labels.items():
            x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
            y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
            text = ax.text(x, y, label, rotation=angle, ha='center', va='center',visible=draw_labels)
            text.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))




def draw_graph(edges,s,t,currents,layout,draw_labels = False):
    G = nx.DiGraph()
    for i in range(len(edges)):
        edge = edges[i]
        if currents[i] > 0:
            G.add_edge(edge[0],edge[1],weight=edges[i][2])
        else:
            G.add_edge(edge[1],edge[0],weight=edges[i][2])
        
        
    if layout == Layout.SPRING:
        pos = nx.spring_layout(G) 
    elif layout == Layout.SPECTRAL:
        pos = nx.spectral_layout(G)
    elif layout == Layout.CIRCULAR:
        pos = nx.circular_layout(G)
    elif layout == Layout.SPIRAL:
        pos = nx.spiral_layout(G)
    else:
        pos = nx.random_layout(G)

     
    node_colors = ['grey' if node != s and node != t else 'green' for node in G.nodes()]
    edge_colors = edge_coloring(edges,G,currents)


    plt.figure(figsize=(12, 6))    

    nx.draw(G, pos,width=3, with_labels=True, node_size=700, node_color=node_colors, font_size=10, font_weight='bold',edge_color=edge_colors, arrows=True)

    edge_labels = edge_labeling(edges,currents)
    draw_edge_labels_with_angle(edge_labels, pos,draw_labels)

    plt.show()


