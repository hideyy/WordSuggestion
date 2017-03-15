import os, sys
import math
import pickle

from gensim.models import word2vec

import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

#wordnet-1.7.1 の読み込み
cwd = os.getcwd()
nltk.data.path.append(cwd)
wordnet17_dir="resources/WordNet-1.7.1/"
wn17_path = "{0}/dict".format(wordnet17_dir)
WN17 = WordNetCorpusReader(os.path.abspath("{0}/{1}".format(cwd, wn17_path)), nltk.data.find(wn17_path))

def tf(terms, document):
    """
    TF値の計算。単語リストと文章を渡す
    :param terms:
    :param document:
    :return:
    """
    tf_values = [document.count(term) for term in terms]
    return list(map(lambda x: x/sum(tf_values), tf_values))


def idf(terms, documents):
    """
    IDF値の計算。単語リストと全文章を渡す
    :param terms:
    :param documents:
    :return:
    """
    return [math.log10(len(documents)/sum([bool(term in document) for document in documents])) for term in terms]


def tf_idf(terms, documents):
    """
    TF-IDF値を計算。文章毎にTF-IDF値を計算
    :param terms:
    :param documents:
    :return:
    """
    return [[_tf*_idf for _tf, _idf in zip(tf(terms, document), idf(terms, documents))] for document in documents]

def create_documents_and_terms():
    documents = []
    terms = []
    i = 0
    for synset in WN17.all_synsets():
        #定義文をglossに格納
        gloss = synset.definition()
        #定義文から";", "(", ")"を削除
        gloss = gloss.replace(";", "").replace("(", "").replace(")", "").replace(":", "").replace('"', "").replace("'", "").lower()
        documents.append(gloss)

        gloss_words = gloss.split(" ")
        for gw in gloss_words:
            if gw not in terms:
                terms.append(gw)

        i += 1
        if i%1000 == 0:
            print("%d Finished" % i)

    return documents, terms

def connect_terms2idfs(terms, idf_list):
    dictionary = {}
    for i in range(len(terms)):
        dictionary[terms[i]] = idf_list[i]

    return dictionary


if __name__ == '__main__':
    # documents, terms = create_documents_and_terms()
    # idf_list = idf(terms, documents)
    # dic = connect_terms2idfs(terms, idf_list)
    # with open('gloss17_idfs.pickle', 'wb') as f:
    #     pickle.dump(dic, f)

    with open('gloss_idfs.pickle', 'rb') as f:
        dic = pickle.load(f)
    print(dic['to'])
    with open('gloss17_idfs.pickle', 'rb') as f:
        idfs = pickle.load(f)

    model = word2vec.Word2Vec.load_word2vec_format("../word2vec/models/GoogleNews-vectors-negative300.bin", binary=True)

    vec_dict = {}
    index = 0

    for synset in WN17.all_synsets():
        vec = 0
        gloss = synset.definition()
        gloss = gloss.replace(";", "").replace("(", "").replace(")", "").replace(":", "").replace('"', "").replace("'", "").lower()

        gloss_words = gloss.split(" ")
        for gw in gloss_words:
            if gw in model.wv.vocab:
                if vec is 0:
                    vec = idfs[gw]*model[gw]
                else:
                    vec += idfs[gw]*model[gw]

        vec_dict[synset.name()] = vec
        index += 1
        if index%100 == 0:
            print("%d Finished!!" % index)

    with open('synset2vecG.pickle', 'wb') as f:
        pickle.dump(vec_dict, f)
