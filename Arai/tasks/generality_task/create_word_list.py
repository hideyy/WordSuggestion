# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import sys
import numpy as np

import pickle

if __name__ == '__main__':
    # 学習済みモデルのロード
    # model = word2vec.Word2Vec.load("models/wn_glosses.model")
    model = word2vec.Word2Vec.load("../word2vec/models/automobile.en.wiki.model")
    # model = word2vec.Word2Vec.load("models/sample.model")

    wordVecs_temp = []
    words = []

    for i in range(1000):
        w = list(model.vocab.keys())[i]
        try:
            print(w)
            words.append(w)
        except:
            continue

    with open('automobile_word_list1000.pickle', 'wb') as f:
        pickle.dump(words, f)
