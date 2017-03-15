# coding: utf-8
import os
import sys
import numpy as np
from gensim.models import word2vec
import logging

import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

from common.util import WordNetModel
from common.util import WVModel
from common.util import VectorMethod

import pickle
from nltk import stem

def synset2vec_glossbase():
    result = {}
    i = 0
    #全てのsynsetに対して
    for synset in wnm.wn.all_synsets():
        #定義文をglossに格納
        gloss = synset.definition()
        #定義文から";", "(", ")"を削除
        gloss = gloss.replace(";", "").replace("(", "").replace(")", "").replace(":", "").replace('"', "").replace("'", "").lower()
        #定義文を空白でsplitしてlistに
        gloss_words = gloss.split(" ")

        #見出し語化
        lemmatizer = stem.WordNetLemmatizer()

        result_vecs = []
        num = 0
        #あるsynsetのglossを構成する単語のベクトルの和をそのsynsetのベクトルとする
        for gw in gloss_words:
            #word2vecのボキャブラリーにある
            if wvm.is_in_vocab(gw):
                temp_vec = wvm.vector(gw)
                result_vecs.append(temp_vec)
                num += 1

        if num is not 0:
            result[synset.name()] = vm.average_vector(result_vecs)

        i += 1
        if (i%1000 == 0):
            logger.info("Saved " + str(i) + " synsets")

    with open('models/synset2vecGAve.pickle', 'wb') as f:
        pickle.dump(result, f)

def synset2vec_wordbase():
    result = {}
    i = 0
    lnum = 0
    #全てのsynsetに対して
    for synset in wnm.wn.all_synsets():
        #Synsetに含まれるwordの取得
        words = []
        for l in synset.lemmas():
            words.append(l.name())
        #word2vecでvocabにあったwordの数を格納
        num = 0
        result_vecs = []
        for w in words:
            #word2vecのボキャブラリーにあるなら
            if wvm.is_in_vocab(w):
                temp_vec = wvm,vector(w)
                result_vecs.append(temp_vec)
                num += 1
        if num is not 0:
            result[synset.name()] = vm.average_vector(result_vecs)
        i += 1
        if (i%1000 == 0):
            logger.info("Saved " + str(i) + " synsets")
    with open('models/synset2vecW.pickle', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    #ログの出力
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    wnm = WordNetModel('1.7.1')
    wvm = WVModel('GoogleNews')
    vm = VectorMethod()

    #synset2vec作成
    synset2vec_wordbase()
    synset2vec_glossbase()
