# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import sys
import numpy as np

from wordnet_affect import wn_affect_class as wa
from wordnet_affect import wn_affect_hierarchy_class as wah

import os
os.environ["NLTK_DATA"] = os.getcwd()
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import *
import nltk
from nltk.corpus import WordNetCorpusReader
from sqlalchemy import *
from xml.dom import minidom
from nltk.corpus import wordnet as wn

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

import pickle
from distutils.util import strtobool


#word2vecで学習済みの単語だけを拾い，ベクトルを並べて行列化
def make_vectors(words):
    result_vecs = []
    result_words = []
    found = 0
    not_found = 0

    for w in words:
        #word2vecのボキャブラリーにあるなら
        if w in model.vocab:
            result_vecs.append(model[w])
            result_words.append(w)
            print("%s found!!" % w)
            found += 1
        else:
            # print("%s NOT found" % w)
            not_found += 1

    result_vecs = np.array(result_vecs)
    print("%d / %d words found" % (found, found+not_found))

    return result_vecs, result_words

#python categolize.py "categ1" "categ2"
if __name__ == '__main__':
    # 学習済みモデルのロード
    # model = word2vec.Word2Vec.load("models/wn_glosses.model")
    model = word2vec.Word2Vec.load("../word2vec/models/en.wiki.2gb.model")
    # model = word2vec.Word2Vec.load("models/sample.model")

    #比較するカテゴリ
    upper_categ = sys.argv[1]
    # with_w = bool(strtobool(sys.argv[4]))

    #wn-affectの読み込み
    wn_affect = wa.WnAffect()
    wn_affect_hierarchy = wah.WnAffectHierarchy()

    if upper_categ not in wn_affect_hierarchy.get_nodes():
        print("%s is not wn-affect category" % categ)
        sys.exit()

    #上位カテゴリから1つ下の下位カテゴリを取得
    categs = wn_affect_hierarchy.find_child(upper_categ)

    #指定のカテゴリ下に属する全てのカテゴリを抽出しそれらカテゴリの属するwordを抽出：result
    results = []
    for categ in categs:
        categ_words = wn_affect_hierarchy.find_children(categ)
        results.append(wn_affect.search_by_categ(categ_words))

    #wordそれぞれのベクトルを並べて行列化：vecs
    vecs = []
    words = []

    for result in results:
        vecs_temp, words_temp = make_vectors(result)
        vecs.append(vecs_temp)
        words.append(words_temp)

    #それぞれのカテゴリのwordベクトルの平均
    vecs_ave = []
    for vec in vecs:
        ave_temp = sum(vec)/len(vec)
        vecs_ave.append(ave_temp)


    result = {}

    for index, categ in enumerate(categs):
        result[categ] = vecs_ave[index]

    with open(upper_categ+'_vectors.pickle', 'wb') as f:
        pickle.dump(result, f)
