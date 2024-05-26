import numpy as np
from prad  import solver,draw_graph,Layout
from graph_generator import generating_random_grid_graph,generating_random_regular_graph,generating_random_bridge_graph,generating_random_graph

def graph_test(s,t,L,draw_labels = False):
    random_edges = [(0,1,3),
        (1,2,2),
        (2,3,1),
        (3,0,4),
        (0,2,2),
        (1,3,2)]        
    currents = solver(random_edges,s,t,3)
    draw_graph(random_edges,s,t,currents,L,draw_labels)

def random_regular_graph_test(v,s,t,L,draw_labels = False):
    random_edges = generating_random_regular_graph(v,0.5)
    currents = solver(random_edges,s,t,20)
    draw_graph(random_edges,s,t,currents,L,draw_labels)

def random_bridge_graph_test(v,s,t,L,draw_labels = False):
    random_edges = generating_random_bridge_graph(v,1,2)
    currents = solver(random_edges,s,t,0.5)
    draw_graph(random_edges,s,t,currents,L,draw_labels)

def random_grid_graph_test(rows,cols,s,t,L,draw_labels = False):
    random_edges = generating_random_grid_graph(rows,cols,0.2)
    currents = solver(random_edges,s,t,1)
    draw_graph(random_edges,s,t,currents,L,draw_labels)

def random_graph_test(v,s,t,L,draw_labels = False):
    random_edges = generating_random_graph(v,1)
    currents = solver(random_edges,s,t,5)
    draw_graph(random_edges,s,t,currents,L,draw_labels)


graph_test(0,3,Layout.SPRING,True)
""" random_regular_graph_test(5,0,1,Layout.CIRCULAR,True) """ 
""" random_grid_graph_test(5,5,0,1,Layout.SPECTRAL) """
""" random_graph_test(7,0,1,Layout.SPRING,True) """
""" random_bridge_graph_test(20,0,1,Layout.SPECTRAL) """

