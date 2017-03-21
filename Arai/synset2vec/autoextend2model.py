# coding: utf-8

import os
import sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.util import WordNetModel

import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import pickle


if __name__ == '__main__':
    wnm = WordNetModel('1.7.1')

    f = open('resource/AutoExtend/synsets.txt')
    line = f.readline()
    lines = []

    while line:
        lines.append(line)
        line = f.readline()
    f.close

    dic = {}

    for idx, l in enumerate(lines[1:]):
        #最後の改行を除いてスペースでスプリット
        temp = l.replace("\n", "").split(" ")
        word = temp[0].split("-")[2:]
        synset = wnm.wn._synset_from_pos_and_offset(word[1], int(word[0]))
        embedding = [float(x) for x in temp[1:]]
        dic[synset.name()] = embedding

        if idx%100 == 0:
            print("%d Finished!!" % idx)

    with open('models/synset2vecAE.pickle', 'wb') as f:
        pickle.dump(dic, f)
