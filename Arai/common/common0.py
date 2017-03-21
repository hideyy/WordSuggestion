import numpy as np
from gensim.models import word2vec

class VectorObject(object):

    def __init__(self, vector):
        self._vector = np.array(vector)

    def length(self):
        return sum(self._vector*self._vector)

    def normalize(self):
        return self._vector/self.length()

    def cos_sim(self, other):
        return sum(self._vector*other._vector)/np.sqrt(self.length()*other.length())

class WordVectorObject(VectorObject):
    def __init__(self, wv_model):
        self._wv_model = wv_model
        self._vector = []

class WVModel(object):
    ModelNameMap = {'GoogleNews': 'GoogleNews-vectors-negative300.bin', 'Wikipedia2GB' : 'en.wiki.2gb.model'}
    def __init__(self, modelname):
        if modelname == 'GoogleNews':
            self._model = word2vec.Word2Vec.load_word2vec_format("../models/"+ModelNameMap[modelname], binary=True)
        else:
            self._model = word2vec.Word2Vec.load("../models/"+ModelNameMap[modelname])]

    def is_in_vocab(word):
        if word in self._model.wv.vocab:
            return True
        else:
            return False

    def vector(word):
        wordvector = WordVectorObject(self)
        wordvector._vector = self._model[word]
        return wordvector
