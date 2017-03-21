import os, sys
import math
import pickle

import numpy as np

from gensim.models import word2vec

import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import csv   #csvモジュールをインポートする
import pandas as pd

from scipy.stats import spearmanr

import re

#wordnet-1.7.1 の読み込み
cwd = os.getcwd()
nltk.data.path.append(cwd)
wordnet17_dir="resources/WordNet-1.7.1/"
wn17_path = "{0}/dict".format(wordnet17_dir)
WN17 = WordNetCorpusReader(os.path.abspath("{0}/{1}".format(cwd, wn17_path)), nltk.data.find(wn17_path))

if __name__ == '__main__':
    mapping = ['AutoExtend', 'GlossTfIdf', 'GlossAve', 'Word']
    S = {}
    N = 999 #datanum

    # 学習済みモデルのロード
    model = word2vec.Word2Vec.load_word2vec_format("../word2vec/models/GoogleNews-vectors-negative300.bin", binary=True)

    with open('synset2vecAE.pickle', 'rb') as f:
        S['AutoExtend'] = pickle.load(f)

    with open('synset2vecG.pickle', 'rb') as f:
        S['GlossTfIdf'] = pickle.load(f)

    with open('synset2vecGAve.pickle', 'rb') as f:
        S['GlossAve'] = pickle.load(f)

    with open('synset2vecW.pickle', 'rb') as f:
        S['Word'] = pickle.load(f)

    df = pd.read_csv('ratings1.csv')

    Lists = {}

    for elem in mapping:
        Lists[elem] = []
        Lists[elem+'_s1'] = []
        Lists[elem+'_s2'] = []

    Temps = {}
    S1 = {}
    S2 = {}

    for i in range(N):
        Temps[elem] = [-1]

        c1 = re.sub(r'[^a-zA-Z ]', '', df['Context1'][i]).lower().split(' ')
        c1_list = []
        for c in c1:
            if c in model.wv.vocab:
                c1_list.append(model[c])
        c1_vec = sum(c1_list)

        c2 = re.sub(r'[^a-zA-Z ]', '', df['Context2'][i]).lower().split(' ')
        c2_list = []
        for c in c2:
            if c in model.wv.vocab:
                c2_list.append(model[c])
        c2_vec = sum(c2_list)
        for elem in mapping:
            sim1 = 0
            # for s1 in WN17.synsets(df['Word1'][i]):
            for s1 in WN17.synsets(df['Word1'][i], pos = df['POS1'][i]):
                if s1.name() in S[elem]:
                    if S[elem][s1.name()] is not 0:
                        temp1 = sum(np.array(S[elem][s1.name()])*c1_vec)/np.sqrt(sum(np.array(S[elem][s1.name()])*np.array(S[elem][s1.name()]))*sum(c1_vec*c1_vec))
                    else:
                        temp1 = 0
                else:
                    temp1 = 0

                if temp1 > sim1:
                    s1c = s1.name()
                    sim1 = temp1
            if sim1 is 0:
                Lists[elem+'_s1'].append('None')
            else:
                Lists[elem+'_s1'].append(s1c)

            sim2 = 0
            # for s2 in WN17.synsets(df['Word2'][i]):
            for s2 in WN17.synsets(df['Word2'][i], pos = df['POS2'][i]):
                if s2.name() in S[elem]:
                    if S[elem][s2.name()] is not 0:
                        temp2 = sum(np.array(S[elem][s2.name()])*c2_vec)/np.sqrt(sum(np.array(S[elem][s2.name()])*np.array(S[elem][s2.name()]))*sum(c2_vec*c2_vec))
                    else:
                        temp2 = 0
                else:
                    temp2 = 0

                if temp2 > sim2:
                    s2c = s2.name()
                    sim2 = temp2
            if sim2 is 0:
                Lists[elem+'_s2'].append('None')
            else:
                Lists[elem+'_s2'].append(s2c)

            if (sim1 == 0) or (sim2 == 0):
                Lists[elem].append(-1)
            else:
                Lists[elem].append(sum(np.array(S[elem][s1c])*np.array(S[elem][s2c]))/np.sqrt(sum(np.array(S[elem][s1c])*np.array(S[elem][s1c]))*sum(np.array(S[elem][s2c])*np.array(S[elem][s2c]))))

        print(i)

    for elem in mapping:
        df[elem] = Lists[elem]
        df[elem+'_s1'] = Lists[elem+'_s1']
        df[elem+'_s2'] = Lists[elem+'_s2']

    df.to_csv( 'scws_raiting_pos1.csv' )

    df = pd.read_csv('scws_raiting_pos1.csv')

    mainList = df['mean']
    for elem in mapping:
        r = spearmanr(mainList, df[elem])
        print(elem)
        print(r)
