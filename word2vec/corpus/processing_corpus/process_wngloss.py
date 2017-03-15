#!/usr/bin/python
# coding: UTF-8
from nltk import stem
from nltk.corpus import stopwords
from nltk import tokenize

from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn


def remove_stopwords(words):
    #文章をlistに変換
    ws = tokenize.wordpunct_tokenize(words)
    #stopword：is, a, etc...
    stopset = set(stopwords.words('english'))
    result = []
    for word in words:
        if len(word) < 3 or word in stopset:
            continue
        # print(word)
        result.append(word)
    return result

def lemmalize_list(word_list):
    lemmatizer = stem.WordNetLemmatizer()
    result = []
    for word in word_list:
        # print(lemmatizer.lemmatize(word))
        result.appned(lemmatizer.lemmatize(word))
    return result


Glosses = ""
i = 0
#WordNetから定義文抽出：lemma is definition
for synset in wn.all_synsets():
    for lemma in synset.lemmas():
        Glosses+=lemma.name()+" is "+synset.definition()+"\n"
        i+=1
        if i%100==0:
            print("%d lemma finished!!" % i)

#Not lemmatize
result = Glosses

#lemmatize

# word_list = remove_stopwords(Glosses)
# result = lemmalize_list(word_list)
# result = " ".join(result)

f = open('wn_glosses.txt', 'w') # 書き込みモードで開く
f.write(result) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる
