# coding: utf-8
import os
import sys
sys.path.append(os.pardir)

import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import pickle

from common.util import WordNetModel

if __name__ == '__main__':
    #synset同士の対義語は存在しないため，2つのsynsetが互いに対義lemmaを持つ時対義synsetとみなす
    anto_synsets = []
    anto_words = []

    wnm = WordNetModel('3.5')

    for ss in wnm.wn.all_synsets():
        for l in ss.lemmas():
            for a in l.antonyms():
                anto_synsets.append((ss.name(), a.synset().name()))
                anto_words.append((l.name(), a.name()))
                # for s in  a.synset().similar_tos():
                #     anto_synsets.append((ss.name(), s.name()))

    with open('anto_synsets_notsim'+wnm.wordnet_version()+'.pickle', 'wb') as f:
        pickle.dump(anto_synsets, f)
    with open('anto_words'+wnm.wordnet_version()+'.pickle', 'wb') as f:
        pickle.dump(anto_words, f)
