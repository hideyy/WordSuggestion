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

#wordnet-1.7.1 の読み込み
cwd = os.getcwd()
nltk.data.path.append(cwd)
wordnet17_dir="resources/WordNet-1.7.1/"
wn17_path = "{0}/dict".format(wordnet17_dir)
WN17 = WordNetCorpusReader(os.path.abspath("{0}/{1}".format(cwd, wn17_path)), nltk.data.find(wn17_path))

if __name__ == '__main__':
    mapping = ['AutoExtend', 'GlossTfIdf', 'GlossAve', 'Word']
    S = {}

    with open('synset2vecAE.pickle', 'rb') as f:
        S['AutoExtend'] = pickle.load(f)

    with open('synset2vecG.pickle', 'rb') as f:
        S['GlossTfIdf'] = pickle.load(f)

    with open('synset2vecGAve.pickle', 'rb') as f:
        S['GlossAve'] = pickle.load(f)

    with open('synset2vecW.pickle', 'rb') as f:
        S['Word'] = pickle.load(f)

    df = pd.read_csv('combined.csv')

    Lists = {}

    for elem in mapping:
        Lists[elem] = []

    Temps = {}
    S1 = {}
    S2 = {}

    for elem in mapping:
        print(elem)
        for i in range(353):
            Temps[elem] = [-1]

            for s1 in WN17.synsets(df['Word 1'][i]):
                S1[elem] = np.array(S[elem][s1.name()])
                for s2 in WN17.synsets(df['Word 2'][i]):
                    S2[elem] = np.array(S[elem][s2.name()])
                    Temps[elem].append(sum(S1[elem]*S2[elem])/np.sqrt(sum(S1[elem]*S1[elem])*sum(S2[elem]*S2[elem])))
            Lists[elem].append(max(Temps[elem]))
            print(i)

        df[elem] = Lists[elem]

    df.to_csv( 'updating2.csv' )

    df = pd.read_csv('updating2.csv')

    mainList = df['Human (mean)']
    for elem in mapping:
        r = spearmanr(mainList, df[elem])
        print(elem)
        print(r)
