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
    model = word2vec.Word2Vec.load("../word2vec/models/automobile.en.wiki.model")
    # model = word2vec.Word2Vec.load("models/sample.model")

    lengths = []
    file_name = sys.argv[1]
    with open(file_name+'.pickle', 'rb') as f:
        words = pickle.load(f)

    for i in range(len(words)):
        w = words[i]
        v = model[w]
        l = np.sqrt(sum(v*v))
        lengths.append(l)

    Dict = {}
    for i in range(len(words)):
        Dict[words[i]] = lengths[i]
    Dict = sorted(Dict.items() , key=lambda x: x[1])

    res = ""
    for i in range(len(Dict)):
        res+=str(i)+","+Dict[i][0]+","+str(Dict[i][1])+"\n"

    f = open(file_name+'_length.txt', 'w')
    f.write(res)
    f.close()
