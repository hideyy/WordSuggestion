# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('../../')
import logging

import numpy as np
from gensim.models import word2vec
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

import pickle

from common.util import WordNetModel
from common.util import WVModel
from common.util import VectorMethod
from common.util import CategoryHierarchyObject

#word2vecで学習済みの単語だけを拾い，ベクトルを並べて行列化
def make_vectors(words, wvm):
    result_vecs = []
    result_words = []
    found = 0
    not_found = 0

    for w in words:
        #word2vecのボキャブラリーにあるなら
        if wvm.is_in_vocab(w):
            result_vecs.append(wvm.vector(w))
            result_words.append(w)
            # print("%s found!!" % w)
            found += 1
        else:
            # print("%s NOT found" % w)
            not_found += 1

    result_vecs = np.array(result_vecs)
    # print("%d / %d words found" % (found, found+not_found))

    return result_vecs, result_words

class CategorizeMethod(VectorMethod):
    # vm = VectorMethod()

    def calc_distance(self, category_vector_list, vector):
        result = []
        for category_vector in category_vector_list:
            temp = self.distance(category_vector, vector)
            result.append(temp)
        return result

    def calc_cosine(self, category_vector_list, vector):
        result = []
        for category_vector in category_vector_list:
            temp = self.cos_sim(category_vector, vector)
            result.append(temp)
        return result

    def calc_angle(self, category_vector_list, vector):
        result = []
        for category_vector in category_vector_list:
            temp = self.cos_sim(category_vector, vector)
            result.append(np.arccos(temp))
        return result

    def calc_projected_distance(self, category_vector_list, vector):
        result = []
        basic_vectors = self.basic_vectors(category_vector_list)
        #categoryベクトルとtarget vectorをcategroyベクトル空間に射影
        projected = self.projection(basic_vectors, basic_vectors+vector)

        basic_category_vector_list = projected[:-1]
        target_vector = projected[-1]

        for basic_category_vector in basic_category_vector_list:
            temp = self.distance(basic_category_vector, target_vector)
            result.append(temp)
        return result

    def calc_projected_cosine(self, category_vector_list, vector):
        result = []
        basic_vectors = self.basic_vectors(category_vector_list)
        #categoryベクトルとtarget vectorをcategroyベクトル空間に射影
        projected = self.projection(basic_vectors, basic_vectors+vector)

        basic_category_vector_list = projected[:-1]
        target_vector = projected[-1]

        for basic_category_vector in basic_category_vector_list:
            temp = self.cos_sim(basic_category_vector, target_vector)
            result.append(temp)
        return result

    def calc_projected_angle(self, category_vector_list, vector):
        result = []
        basic_vectors = self.basic_vectors(category_vector_list)
        #categoryベクトルとtarget vectorをcategroyベクトル空間に射影
        projected = self.projection(basic_vectors, basic_vectors+vector)

        basic_category_vector_list = projected[:-1]
        target_vector = projected[-1]

        for basic_category_vector in basic_category_vector_list:
            temp = self.cos_sim(basic_category_vector, target_vector)
            result.append(np.arccos(temp))
        return result

class CategoryVectorObject(VectorMethod):
    vm = VectorMethod()
    def __init__(self, category_hierarchy_object, word_vector_model ,category_name):
        self._category_hierarchy_object = category_hierarchy_object
        self._word_vector_model = word_vector_model
        #入力categoryに一致するcategoryがない場合
        if category_name not in self._category_hierarchy_object.get_nodes():
            print("%s is not category" % category_name)
            sys.exit()
        self.name = category_name
        #categoryに属する単語を取得
        temp_words = self._category_hierarchy_object.find_all_descendants(category_name)
        #categoryに属する単語の中でword2vecに属する単語を抽出し，ベクトルを並列化
        self.vectors, self.words = make_vectors(temp_words, self._word_vector_model)
        #categoryの平均ベクトル
        self.ave_vector = self.average_vector(self.vectors)

class MainCategoryVectorObject:
    def __init__(self, category_hierarchy_object, word_vector_model, main_category_name):
        self._category_hierarchy_object = category_hierarchy_object
        self._word_vector_model = word_vector_model

        if main_category_name not in self._category_hierarchy_object.get_nodes():
            print("%s is not category" % category_name)
            sys.exit()
        self.name = main_category_name
        #main category下1階層のカテゴリリストを取得
        self.category_names = self._category_hierarchy_object.find_children(main_category_name)
        #category objectの配列
        self.category = []
        for cn in self.category_names:
            try:
                temp = CategoryVectorObject(self._category_hierarchy_object, self._word_vector_model, cn)
                self.category.append(temp)
            except:
                continue

    def judge_category(self, method, word_vectors):
        category_vector_list = []
        result_data = []
        result_category = []
        for category in self.category:
            category_vector_list.append(category.ave_vector)
        for wv in word_vectors:
            temp = method(category_vector_list, wv)
            #最大値を取るindexの取得
            max_index = temp.index(max(temp))
            most_likely_category = self.category[max_index].name
            result_data.append(temp)
            result_category.append(most_likely_category)
        return result_data, result_category

    def output_csv_file(self, word_list, result_data, result_category):
        df = pd.DataFrame({'word', word_list})
        result_data = np.array(result_data).T
        df['result'] = result_category
        for index, category in enumerate(self.category):
            df[category.name()] = result_data[index]
        df.to_csv('result/'+self.name+'.csv')

if __name__ == '__main__':

    wnm = WordNetModel('1.7.1')
    wvm = WVModel('Wikipedia2GB')
    cm = CategorizeMethod()

    #比較するカテゴリ/入力値から取得
    # categs = [sys.argv[1], sys.argv[2]]

    #wn-affectのhierarchy object
    cho = CategoryHierarchyObject('../../wordnet_affect/wn_affect.pickle', '../../wordnet_affect/wn_affect_hierarchy.pickle')

    cvo = MainCategoryVectorObject(cho, wvm, 'positive-emotion')

    word_list = ['happy', 'positive', 'affective']

    #word listのベクトル化
    word_vectors = []
    for w in word_list:
        word_vectors.append(wvm.vector(w))

    result_data, result_category = cvo.judge_category(cm.calc_angle, word_vectors)
    print(result_data)
    print(result_category)

    # cvo.output_csv_file(word_list, result_data, result_category)
