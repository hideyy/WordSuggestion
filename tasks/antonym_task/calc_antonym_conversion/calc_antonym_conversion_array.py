# coding: utf-8
import os
import sys
sys.path.append('../../')
import numpy as np
import pickle
from gensim.models import word2vec

from common.util import WVModel

def stat(o, e):
    e = np.vstack([np.ones(e.shape[1]), e])
    return np.linalg.lstsq(e.T, o)[0]


class AntonymConversion:
    def __init__(self, wv_model_name, seed_set):
        self.wv_model_name = wv_model_name
        self.wvm = WVModel(wv_model_name)
        self.seed_set = seed_set
        with open('seed_set/'seed_set, 'rb') as f:
            self.antonyms = pickle.load(f)

    def vectorize(self):
        x = []
        y = []
        for antonym in self.antonyms:
            w1 = antonym[0]
            w2 = antonym[1]
            if self.wvm.is_in_vocab(w1) and self.wvm.is_in_vocab(w2):
                x.append(self.wvm.vector(w1))
                y.append(self.wvm.vector(w2))
        return x,y

    def calc_conversion_array(self, x, y):
        dataX = np.array(x).T
        dataY = np.array(y).T

        W = []
        b = []

        for i in range(len(dataY)):
            obj = dataY[i]
            exp = dataX
            temp = stat(obj, exp)
            W.append(temp[1:])
            b.append(temp[0])

        with open('../antonym_conversion/'+self.seed_set+'_'+self.wvm.model_name+'_W.pickle', 'wb') as f:
            pickle.dump(W, f)
        with open('../antonym_conversion/'+self.seed_set+'_'+self.wvm.model_name+'_b.pickle', 'wb') as f:
            pickle.dump(b, f)

        return W, b

if __name__ == '__main__':
    ac = AntonymConversion('Wikipedia2GB', 'anto_words.pickle')
    x,y = ac.vectorize()
    W,b = ac.calc_conversion_array(x,y)
