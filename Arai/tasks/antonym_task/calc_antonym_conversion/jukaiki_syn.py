# coding: utf-8
import os
import sys
import numpy as np
import pickle
from gensim.models import word2vec
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

def stat(o, e):
    e = np.vstack([np.ones(e.shape[1]), e])
    return np.linalg.lstsq(e.T, o)[0]

def s2v(synset, dic):
    if synset.name() in dic:
        return dic[synset.name()]
    else:
        return 0

def similar_synset(vec, pos, dic):
    result = {}

    for k,v in dic.items():
        if v is not 0:
            if wn.synset(k).pos() in pos:
                v_nmlz = v/np.sqrt(sum(v*v))
                vec_nmlz = vec/np.sqrt(sum(vec*vec))
                res = v_nmlz.dot(vec_nmlz)
                result[k] = res

    result = sorted(result.items() , key=lambda x: x[1], reverse=True)

    return result

def show_ranking(dic, num):
    for i in range(num):
        print("%d : %s %f" % (i+1, dic[i][0], dic[i][1]))


# Dataset作成
# 対義語のペアデータインポート
with open('anto_synsets_notsim.pickle', 'rb') as f:
    anto_synsets = pickle.load(f)
with open('wn_w2v_2.pickle', 'rb') as f:
    dic = pickle.load(f)

# x = []
# y = []
# print(len(anto_synsets))
# i = 0
# for anto_synset in anto_synsets:
#     i += 1
#     syn1 = wn.synset(anto_synset[0])
#     syn2 = wn.synset(anto_synset[1])
#     if (s2v(syn1, dic) is not 0) and (s2v(syn2, dic) is not 0):
#         x.append(s2v(syn1, dic))
#         y.append(s2v(syn2, dic))
#     if i%1000 == 0:
#         print(i)
#
# dataX = np.array(x).T
# dataY = np.array(y).T
#
# with open('antsynsets_datasetXga2.pickle', 'wb') as f:
#     pickle.dump(dataX, f)
# with open('antsynsets_datasetYga2.pickle', 'wb') as f:
#     pickle.dump(dataY, f)

# 重回帰行列計算
# with open('antsynsets_datasetXga2.pickle', 'rb') as f:
#     dataX = pickle.load(f)
# with open('antsynsets_datasetYga2.pickle', 'rb') as f:
#     dataY = pickle.load(f)
#
# W = []
# b = []
#
# for i in range(len(dataY)):
#     obj = dataY[i]
#     exp = dataX
#     temp = stat(obj, exp)
#     W.append(temp[1:])
#     b.append(temp[0])
#
# print(b)
#
# with open('antsynsets_resWga2.pickle', 'wb') as f:
#     pickle.dump(W, f)
# with open('antsynsets_resbga2.pickle', 'wb') as f:
#     pickle.dump(b, f)


with open('antsynsets_resWga2.pickle', 'rb') as f:
    W = pickle.load(f)
with open('antsynsets_resbga2.pickle', 'rb') as f:
    b = pickle.load(f)

W = np.array(W)
b = np.array(b)

syn = wn.synset('funny.a.01')
x = s2v(syn, dic)

x = np.array(x)

res = W.dot(x) + b

dic = similar_synset(res, ['a', 's'], dic)
show_ranking(dic, 10)
