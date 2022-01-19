#!/usr/bin/python

##
# Initial graphical causal discovery using causallearn package
# @author Neil Munjal

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.FCMBased import lingam
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci
import networkx as nx
import pandas as pd
import numpy as np

file = '/home/repositories/git/nkmj/topicc-local/data/causal/neuro_cont_mort_nona.csv'
# requires simple flat file of all continuous physiologic variables without any
# missing values, created separately and exported to the above file. Below is
# the same file with missing values kept in.
df = pd.read_csv(file)
cg = pc(df.to_numpy(),0.025, fisherz,True,0,-1,False)
#cg.draw_pydot_graph()

def label_graph(G,columns):
    for n,col in zip(G.nodes,columns):
        n.set_name(col)
    return G
label_graph(cg.G,df.columns)

cg.to_nx_graph()
#cg.draw_nx_graph()

label_conversion = {i:val for i,val in enumerate(df.columns.to_list())}

g_to_be_drawn = cg.nx_graph
edges = g_to_be_drawn.edges()
colors = [g_to_be_drawn[u][v]['color'] for u, v in edges]
pos = nx.circular_layout(g_to_be_drawn)
nx.draw(g_to_be_drawn, pos=pos, with_labels=True, labels=label_conversion, edge_color=colors)


# MV-PC
file2 = '/home/repositories/git/nkmj/topicc-local/data/causal/neuro_cont_mort_full.csv'
df2 = pd.read_csv(file2)
cg = pc(df2.to_numpy(),0.01, mv_fisherz,True,0,-1,True)
#cg.draw_pydot_graph()

label_graph(cg.G,df2.columns)

cg.to_nx_graph()
#cg.draw_nx_graph()

label_conversion = {i:val for i,val in enumerate(df2.columns.to_list())}

g_to_be_drawn = cg.nx_graph
edges = g_to_be_drawn.edges()
colors = [g_to_be_drawn[u][v]['color'] for u, v in edges]
pos = nx.circular_layout(g_to_be_drawn)
nx.draw(g_to_be_drawn, pos=pos, with_labels=True, labels=label_conversion, edge_color=colors)


# FCI
# didn't terminate
#G = fci(df.to_numpy(),fisherz,0.01,1)

# LiNGAM
model = lingam.ICALiNGAM(41)
model.fit(df.to_numpy())
G = nx.convert_matrix.from_numpy_matrix(model.adjacency_matrix_.transpose(),create_using=nx.DiGraph)
pos = nx.circular_layout(G)
nx.draw(G, pos=pos, with_labels=True, labels=label_conversion)
