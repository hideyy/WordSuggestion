import os
from os import path
import sys


import numpy as np
#word2vec
from gensim.models import word2vec
#wordnet
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn
#pickle
import pickle
#networkx : 有向グラフ
import networkx as nx

ROOT_PATH = 'C:/research'

#ベクトルの基礎計算
class VectorMethod:

    def length(self, v):
        return np.sqrt(sum(v*v))

    def normalize(self, vector):
        return vector/self.length(vector)

    def cos_sim(self, vector1, vector2):
        return sum(vector1*vector2)/(self.length(vector1)*self.length(vector2))

    def average_vector(self, vectors):
        return sum(vectors)/len(vectors)

    def distance(self, vector1, vector2):
        return self.length(vector1-vector2)

    def basic_vectors(self, vectors):
        basic_vectors = []
        for vector in vectors:
            temp_vector = vector
            #シュミッドの正規直行化
            for bv in basic_vectors:
                temp_vector -= sum(bv*vector)*bv
            basic_vectors.append(self.normalize(temp_vector))
        return basic_vectors

    #ベクトル群をbasic_vectorの張る空間に射影
    def projection(self, basic_vectors, vectors):
        result = []
        for vector in vectors:
            temp_vector = []
            for bv in basic_vectors:
                temp_vector.append(sum(bv, vector))
            result.append(np.array(temp_vector))
        return result

#Word2vecのモデル読み込み
class WVModel:
    ModelNameMap = {'GoogleNews': 'GoogleNews-vectors-negative300.bin', 'Wikipedia2GB' : 'en.wiki.2gb.model'}
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'GoogleNews':
            self._model = word2vec.Word2Vec.load_word2vec_format(path.join(ROOT_PATH, "word2vec/models/"+self.ModelNameMap[model_name]), binary=True)
        else:
            self._model = word2vec.Word2Vec.load(path.join(ROOT_PATH, "word2vec/models/"+self.ModelNameMap[model_name]))

    def is_in_vocab(self, word):
        if word in self._model.wv.vocab:
            return True
        else:
            return False

    def vector(self, word):
        if self.is_in_vocab(word):
            wordvector = self._model[word]
        else:
            print("%s is not in vocab" % word)
            wordvector = []
        return wordvector

#Synset2vecのモデル読み込み
class SVModel:
    ModelNameMap = {'WordAve': 'synset2vecW.pickle', 'GlossAve' : 'synset2vecGAve.pickle', 'GlossTfidf' : 'synset2vecG.picke', 'AutoExtend' : 'synset2vecAE.pickle'}
    def __init__(self, modelname):
        with open(path.join(ROOT_PATH, 'synset2vec/models/'+self.ModelNameMap[modelname]), 'rb') as f:
            self._model = pickle.load(f)

    def is_in_vocab(self, synset):
        if synset.name() in self._model:
            return True
        else:
            return False

    def vector(self, synset):
        if self.is_in_vocab(synset):
            synsetvector = self._model[synset.name()]
        else:
            print("%s is not in vocab" % synset.name())
            synsetvector = []
        return synsetvector

#WordNetのVersion対応
class WordNetModel:
    def __init__(self, wordnet_version = '3.5'):
        self.wordnet_version = wordnet_version
        if self.wordnet_version == '3.5':
            self.wn = wn
        else:
            nltk.data.path.append(ROOT_PATH)
            wn_dir="wordnet/resources/WordNet-"+self.wordnet_version+'/'
            wn_path = "{0}/dict".format(wn_dir)
            self.wn = WordNetCorpusReader(os.path.abspath("{0}/{1}".format(ROOT_PATH, wn_path)), nltk.data.find(wn_path))

#word-category-hierarchyを繋ぐオブジェクト
class CategoryHierarchyObject:
    def __init__(self, category_filename, hierarchy_filename):
        #{word : category}形式の辞書
        with open(category_filename, 'rb') as f:
            self._dict = pickle.load(f)
        #hierarchyの有向グラフ
        with open(hierarchy_filename, 'rb') as f:
            self._G = pickle.load(f)

    def search_by_categ(self, categ_words):
        result = []
        for categ_word in categ_words:
            for k,v in self._dict.items():
                if v==categ_word:
                    result.append(k)
        return result

    def search_by_word(self, word):
        result = []
        for k,v in self._dict.items():
            if k==word:
                result.append(v)
        return result

    def get_nodes(self):
        return self._G.nodes()

    def find_all_descendants(self, categ_word):
        result = []
        temps = [categ_word]
        while len(temps) is not 0:
            nextTemps = []
            for temp in temps:
                keys = list(self._G[temp].keys())
                nextTemps.extend(keys)
            result.extend(nextTemps)
            temps = nextTemps
        return result

    def find_descendants(self, categ_word, num):
        result = []
        temps = [categ_word]
        for i in range(num):
            nextTemps = []
            for temp in temps:
                keys = list(self._G[temp].keys())
                nextTemps.extend(keys)
            result.extend(nextTemps)
            temps = nextTemps
        return result

    def find_children(self, categ_word):
        return list(self._G[categ_word].keys())

#Test Field
# vm = VectorMethod()
# wvm = WVModel('Wikipedia2GB')
# v1 = wvm.vector('hot')
# v2 = wvm.vector('cold')
# print(vm.distance(v1,v2))
# wnm = WordNetModel('1.7.1')
# print(wnm.wn.synsets('hot'))
# cho = CategoryHierarchyObject('../wordnet_affect/wn_affect.pickle', '../wordnet_affect/wn_affect_hierarchy.pickle')
# print(cho.find_all_descendants('neutral-emotion'))
