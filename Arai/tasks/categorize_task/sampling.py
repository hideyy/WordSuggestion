# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('../../')
import logging

import numpy as np
from gensim.models import word2vec
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

import pickle

from common.util import *
from categorize import *

def plot2D(cvo1, cvo2):
    vm = VectorMethod()
    basic_vectors = vm.basic_vectors([cvo1.ave_vector(), cvo2.ave_vector()])
    xy1 = vm.projection(basic_vectors, cvo1.vectors())
    xy2 = vm.projection(basic_vectors, cvo2.vectors())
    x1 = np.array(xy1).T[0]
    y1 = np.array(xy1).T[1]
    x2 = np.array(xy2).T[0]
    y2 = np.array(xy2).T[1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x1,y1, s=15, c='red', marker="o", label=cvo1.name())
    ax.scatter(x2,y2, s=15, c='blue', marker="o", label=cvo2.name())
    ax.legend(loc='upper left')
    plt.savefig(cvo1.name()+'_'+cvo2.name()+'.png' )
    plt.show()

def plot3D(cvo0, cvo1, cvo2):
    vm = VectorMethod()
    basic_vectors = vm.basic_vectors([cvo0.ave_vector(), cvo1.ave_vector(), cvo2.ave_vector()])
    xyz0 = vm.projection(basic_vectors, cvo0.vectors())
    xyz1 = vm.projection(basic_vectors, cvo1.vectors())
    xyz2 = vm.projection(basic_vectors, cvo2.vectors())
    x0 = np.array(xyz0).T[0]
    y0 = np.array(xyz0).T[1]
    z0 = np.array(xyz0).T[2]
    x1 = np.array(xyz1).T[0]
    y1 = np.array(xyz1).T[1]
    z1 = np.array(xyz1).T[2]
    x2 = np.array(xyz2).T[0]
    y2 = np.array(xyz2).T[1]
    z2 = np.array(xyz2).T[2]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x0,y0,z0, s=15, c='red', marker="o", label=cvo0.name())
    ax.scatter(x1,y1,z1, s=15, c='blue', marker="o", label=cvo1.name())
    ax.scatter(x2,y2,z2, s=15, c='green', marker="o", label=cvo2.name())
    ax.legend(loc='upper left')
    plt.savefig(cvo0.name()+'_'+cvo1.name()+'_'+cvo2.name()+'.png' )
    plt.show()

#python sampling.py "main_categ" "sub_categ1" "sub_categ2" true(or false)
if __name__ == '__main__':

    wnm = WordNetModel('1.7.1')
    wvm = WVModel('Wikipedia2GB')
    cm = CategorizeMethod()

    cho = CategoryHierarchyObject('../wordnet_affect/wn_affect.pickle', '../wordnet_affect/wn_affect_hierarchy.pickle')
    cvo1 = CategoryVectorObject(cho, wvm, 'positive-emotion')
    cvo2 = CategoryVectorObject(cho, wvm, 'negative-emotion')

    plot2D(cvo1, cvo2)
