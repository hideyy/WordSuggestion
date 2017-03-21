# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import sys
from distutils.util import strtobool
import numpy as np
import pickle

if __name__ == '__main__':
    # 学習済みモデルのロード
    # model = word2vec.Word2Vec.load("models/wn_glosses.model")
    model = word2vec.Word2Vec.load("models/en.wiki.2gb.model")
    # model = word2vec.Word2Vec.load("models/sample.model")

    wvecs = []
    nwvecs = []
    wvec_lengths = []

    file_name = sys.argv[1]
    with open(file_name+'.pickle', 'rb') as f:
        words = pickle.load(f)

    for i in range(len(words)):
        w = words[i]
        #正規化
        v_temp = model[w]
        l = np.sqrt(sum(v_temp*v_temp))
        v = v_temp/(l*l)
        nv = v_temp/l

        wvecs.append(v)
        nwvecs.append(nv)
        # wvec_lengths.appned(l)

    wvecs = np.array(wvecs)
    nwvecs = np.array(nwvecs)
    relvecs = nwvecs.dot(wvecs.T)
    # for i in range(len(wvecs)):
    #     for j in range(len(wvecs)):
    #         temp = wvecs[i].dot(wvecs[j])/(wvec_lengths[i]*wvec_lengths[j])
    #         relvecs[i][j] = temp
    result = []
    for relvec in relvecs:
        result.append(np.var(relvec))

    Dict = {}
    for i in range(len(words)):
        Dict[words[i]] = result[i]
    Dict = sorted(Dict.items() , key=lambda x: x[1])

    #n, categ, word, pointでtxtファイル出力
    res = ""
    for i in range(len(Dict)):

        res+=str(i)+","+Dict[i][0]+","+str(Dict[i][1])+"\n"

    f = open(file_name+'_generality_3.txt', 'w')
    f.write(res)
    f.close()
