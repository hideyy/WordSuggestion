# coding: utf-8
import os
import sys

#for read xml file
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import *
from sqlalchemy import *
from xml.dom import minidom

import networkx as nx
import matplotlib.pylab as plt
from graphviz import Digraph
import pickle


#Wordnet-Affect hierarchyを読み込んでnetworkxのオブジェクト化
# corpus: a-hierarchy.xml
def load_hierarchy(corpus):
    G = nx.DiGraph()
    tree = ET.parse(corpus)
    root = tree.getroot()
    for elem in root.findall(".//categ"):
        G.add_node(elem.get("name"))

    for elem in root.findall(".//categ"):
        G.add_edge(elem.get("isa"), elem.get("name"))

    return G

#wn-affectの階層をグラフ化
def make_hierarchy_graph(corpus):
    G = Digraph(format="png")

    tree = ET.parse(corpus)
    root = tree.getroot()
    for elem in root.findall(".//categ"):
        G.node(str(elem.get("name")), str(elem.get("name")))

    for elem in root.findall(".//categ"):
        G.edge(str(elem.get("isa")), str(elem.get("name")))

    G.render('wn_affect')

#wn-affectのグラフを可視化
def show_graph(G):
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="w")
    nx.draw_networkx_edges(G, pos, width=1)
    # nx.draw_networkx_edge_labels(G, pos,edge_labels)
    nx.draw_networkx_labels(G, pos ,font_size=5, font_color="r")

    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':

    #wn_affect_hierarchyをpickleファイル化
    G = load_hierarchy("resources/wn-affect-1.1/a-hierarchy.xml")
    with open('wn_affect_hierarchy.pickle', 'wb') as f:
        pickle.dump(G, f)

    #wn_affect_hierarchyをpngファイル化
    make_hierarchy_graph("resources/wn-affect-1.1/a-hierarchy.xml")
