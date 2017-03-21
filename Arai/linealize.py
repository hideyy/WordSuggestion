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

def calc_eigenvector(vecs):
    #vecsの転置行列の共分散行列を求める
    S = np.cov(vecs.T, bias=1)
    #共分散行列の固有値と固有ベクトルを求める
    la, vs = np.linalg.eigh(S)
    evs = []
    #固有ベクトルの正規化
    for v in vs:
        evs.append(v/np.sqrt(np.sum(v*v)))

    return la, evs

def find_axis(evs, vecs):
    #key:固有ベクトル, value: 固有ベクトル上に射影した時の分散値
    d = []
    for ev in evs:
        #固有ベクトルと言語ベクトルの積
        ls = vecs.dot(ev)
        #上記で求めた積の分散値
        lsV = np.var(ls)
        d.append((ev, lsV))
    #valueの小さい順にソート
    # d = sorted(d, key=lambda x: x[1])
    return d

def plot_to_2d(categ, vec1, vec2, vecs, words, color="blue"):
    x = []
    y= []

    for idx, v in enumerate(vecs):
        x.append(v.dot(vec1))
        y.append(v.dot(vec2))
        # ax.annotate(words[idx],xy=(v.dot(vec1),v.dot(vec2)),size=10)

    ax.scatter(x,y, s=10, c=color, marker="+", label=categ)
    # x_ave = sum(x)/len(x)
    # y_ave = sum(y)/len(y)
    # ax.scatter(x_ave,y_ave, s=20, marker=".", label=categ+" average")

# def find_best_axis(d1, d2, vecs1, vecs2, N):
#     evs = []
#     for i in range(N):
#         vecs.append(d1[i][0])
#         vecs.append(d2[i][0])
#     V1s = []
#     V2s = []
#     syaei = []
#     for ev in evs:
#         ls1 = vecs1.dot(ev)
#         ave1 = sum(ls1)/len(ls1)
#         ls2 = vecs2.dot(ev)
#         ave2 = sum(ls2)/len(ls2)
#
#         syaei.append(ave1-ave2)
#
#         lsV1 = np.var(ls1)
#         lsV2 = np.var(ls2)
#         V1s.append(lsV1)
#         V2s.append(lsV2)





if __name__ == '__main__':
    # 学習済みモデルのロード
    # model = word2vec.Word2Vec.load("models/wn_glosses.model")
    model = word2vec.Word2Vec.load("models/en.wiki.2gb.model")
    # model = word2vec.Word2Vec.load("models/sample.model")

    #比較するカテゴリ
    categs = [sys.argv[1], sys.argv[2]]

    #wn-affectの読み込み
    wn_affect = wa.WnAffect()
    wn_affect_hierarchy = wah.WnAffectHierarchy()

    for categ in categs:
        if categ not in wn_affect_hierarchy.get_nodes():
            print("%s is not wn-affect category" % categ)
            sys.exit()

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

    #平均ベクトルと平均ベクトルを繋ぐベクトル
    v1 = vecs_ave[0] - vecs_ave[1]
    # v2 = vecs_ave[1] - vecs_ave[2]
    nv1 = v1/np.sqrt(sum(v1*v1))
    # nv2 = v2/np.sqrt(sum(v2*v2))

    # las = []
    # evs = []
    # for vec in vecs:
    #     la_temp, ev_temp = calc_eigenvector(vec)
    #     las.append(la_temp)
    #     evs.appned(ev_temp)
    # ds = []
    # for i in range(len(vecs)):
    #     ds[i] = find_axis(evs[i], vecs[i])

    # vec1 = nv1
    # vec2 = nv2

    #ベクトル射影して値順にソート
    Dict = {}
    for i in range(len(vecs[0])):
        Dict[words[0][i]] = vecs[0][i].dot(nv1), categs[0]
    for i in range(len(vecs[1])):
        Dict[words[1][i]] = vecs[1][i].dot(nv1)
    Dict = sorted(Dict.items() , key=lambda x: x[1])

    #n, categ, word, pointでtxtファイル出力
    res = ""
    for i in range(len(Dict)):
        res+=str(i)+","+wn_affect.search_by_word(Dict[i][0])+","+Dict[i][0]+","+str(Dict[i][1])+"\n"

    f = open(categs[0]+'_'+categs[1]+'.txt', 'w')
    f.write(res)
    f.close()

    # fig = plt.figure()
    #
    # ax = fig.add_subplot(1,1,1)
    #
    # plot_to_2d(categs[0], vec1, vec2, vecs[0], words[0])
    # plot_to_2d(categs[1], vec1, vec2, vecs[1], words[1], color="red")
    # plot_to_2d(categs[2], vec1, vec2, vecs[2], words[2], color="green")
    #
    # ax.legend(loc='upper left')
    # plt.savefig( 'foo.png' )
    # plt.show()
