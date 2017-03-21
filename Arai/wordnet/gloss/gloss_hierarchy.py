# coding: utf-8
import os
import sys

import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import networkx as nx
import matplotlib.pylab as plt
from graphviz import Digraph
import pickle


#Wordnet-Affect hierarchyを読み込んでnetworkxのオブジェクト化
# corpus: a-hierarchy.xml
def load_hierarchy():
    i = 0
    nodes = []
    G = nx.DiGraph()
    for synset in wn.all_synsets():
        #定義文をglossに格納
        gloss = synset.definition()
        #定義文から";", "(", ")"を削除
        gloss = gloss.replace(";", "").replace("(", "").replace(")", "").replace(":", "").replace('"', "").replace("'", "")
        #定義文を空白でsplitしてlistに
        gloss_words = gloss.split(" ")
        for l in synset.lemmas():
            # if l.name() not in nodes:
            #     nodes.append(l.name())
            #     G.add_node(l.name())
            for gw in gloss_words:
                # if gw not in nodes:
                #     nodes.append(gw)
                #     G.add_node(gw)
                G.add_edge(str(l.name()).lower(), str(gw).lower())
        i += 1
        if i%100 == 0:
            print(i)

    return G

#wn-affectの階層をグラフ化
def make_hierarchy_graph():
    G = Digraph(format="png")
    # node = []
    i = 0

    for synset in wn.all_synsets():
        #定義文をglossに格納
        gloss = synset.definition()
        #定義文から";", "(", ")"を削除
        gloss = gloss.replace(";", "").replace("(", "").replace(")", "").replace(":", "").replace('"', "").replace("'", "")
        #定義文を空白でsplitしてlistに
        gloss_words = gloss.split(" ")
        for l in synset.lemmas():
            # if l.name() not in node:
            #     node.append(l.name())
            #     G.node(l.name(), l.name())
            for gw in gloss_words:
                # if gw not in node:
                #     node.append(gw)
                #     G.node(gw, gw)
                G.edge(str(l.name()).lower(), str(gw).lower())
        i += 1
        if i==50:
            print("pre-finish")
            G.render('gloss_hierarchy')
            sys.exit()

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

    #gloss_hierarchyをpickleファイル化
    G = load_hierarchy()
    with open('gloss_hierarchy.pickle', 'wb') as f:
        pickle.dump(G, f)

    #wgloss_hierarchyをpngファイル化
    # make_hierarchy_graph()
