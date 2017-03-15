import os, sys
import math
import pickle

import numpy as np

from gensim.models import word2vec

import csv   #csvモジュールをインポートする
import pandas as pd

def cos_sim(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return sum(v1*v2)/np.sqrt(sum(v1*v1)*sum(v2*v2))

if __name__ == '__main__':
    mapping = ['res1', 'res2', 'res3', 'res4', 'res5', 'result']
    N =790  #datanum

    # 学習済みモデルのロード
    # model = word2vec.Word2Vec.load("../word2vec/models/en.wiki.2gb.model")
    model = word2vec.Word2Vec.load_word2vec_format("../word2vec/models/GoogleNews-vectors-negative300.bin", binary=True)

    #antonym変換行列のW,bの読み込み
    with open('antwords_resWg.pickle', 'rb') as f:
        W = np.array(pickle.load(f))
    with open('antwords_resbg.pickle', 'rb') as f:
        b = np.array(pickle.load(f))
    #テストセットの読み込み
    df = pd.read_csv('antonym_testset.csv')

    Lists = {}

    for elem in mapping:
        Lists[elem] = []

    count = 0

    for i in range(N):
        cvs = []
        if df['word'][i] in model.wv.vocab:
            wv = np.array(model[df['word'][i]])
            ant_wv = W.dot(wv) + b
            temp_index = -1
            temp_max = -1
            for j in range(5):
                if df['choice'+str(j+1)][i] in model.wv.vocab:
                    temp = cos_sim(ant_wv, model[df['choice'+str(j+1)][i]])
                    Lists['res'+str(j+1)].append(temp)
                    if temp > temp_max:
                        temp_max = temp
                        temp_index = j
                else:
                    Lists['res'+str(j+1)].append(-1)

            if temp_index == -1:
                Lists['result'].append('None')
            else:
                Lists['result'].append(df['choice'+str(temp_index+1)][i])
                print(df['choice'+str(temp_index+1)][i])
                if df['choice'+str(temp_index+1)][i] == df['answer'][i]:
                    count += 1
        else:
            for j in range(5):
                Lists['res'+str(j+1)].append(-1)
            Lists['result'].append('None')

        print(i)

    for elem in mapping:
        print(elem)
        print(len(Lists[elem]))

    for elem in mapping:
        df[elem] = Lists[elem]

    df.to_csv( 'antonym_resultG.csv' )

    print("Number of error : %d" % Lists['result'].count('None'))
    print("Number of correct answer : %d" % count)
