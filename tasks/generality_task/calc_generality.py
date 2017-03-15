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

    wordVecs_temp = []
    file_name = sys.argv[1]
    with open(file_name+'.pickle', 'rb') as f:
        words = pickle.load(f)

    if bool(strtobool(sys.argv[2])):
        string = "_normalized"
        for i in range(len(words)):
            w = words[i]
            #正規化
            v_temp = model[w]
            v = v_temp/np.sqrt(sum(v_temp*v_temp))
            wordVecs_temp.append(v)
    else:
        string = ""
        for i in range(len(words)):
            w = words[i]
            #正規化なし
            wordVecs_temp.append(model[w])


    # for w in model.vocab:
    #     wordVecs_temp.append(model[w])
    #     words.append(w)

    wordVecs = np.array(wordVecs_temp)
    # relationVecs = []
    # for i in range(len(wordVecs)):
    #     temp = wordVecs[i].dot(wordVecs.T)
    #     print(list(temp))
    #     relationVecs.append(list(temp))
    relationVecs = wordVecs.dot(wordVecs.T)

    result = []
    for relationVec in relationVecs:
        result.append(np.var(relationVec))

    Dict = {}
    for i in range(len(words)):
        Dict[words[i]] = result[i]
    Dict = sorted(Dict.items() , key=lambda x: x[1])

    #n, categ, word, pointでtxtファイル出力
    res = ""
    for i in range(len(Dict)):

        res+=str(i)+","+Dict[i][0]+","+str(Dict[i][1])+"\n"

    f = open(file_name+'_generality'+string+'.txt', 'w')
    f.write(res)
    f.close()
