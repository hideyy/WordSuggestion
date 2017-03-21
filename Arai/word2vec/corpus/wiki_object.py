# coding: utf-8

import urllib
import urllib.parse
import urllib.request
# import urllib2
from xml.dom.minidom import parse as parseXML

from gensim.corpora import WikiCorpus

from gensim.corpora.wikicorpus import filter_wiki

import re

class WikiObject(object):
    URLMApping = {'japanese': 'http://ja.wikipedia.org/w/api.php?' , 'english' : 'http://en.wikipedia.org/w/api.php?' }
    BASIC_PARAMETERS = {'action': 'query','format': 'xml'}
    def __init__(self, language = "english"):
        self.language = language
        url = URLMapping[language]
        self._url = url if url.endswith('?') else url + '?'

    def find_child(self, category_name):
        result = []
        parameters = {'list': 'categorymembers',
                      'cmlimit': 500,
                      'cmtitle': category_name}
        page = WikiHandler(parameters, url = self._url)
        elelist = page.dom.getElementsByTagName('cm')
        for ele in elelist:
            result.append(ele.getAttribute('title').encode('sjis', 'ignore').decode('sjis', 'ignore'))
        return result

    def find_article(self, title):
        result = []
        parameters = {'prop': 'revisions',
                      'rvprop': 'content',
                      'titles': title}
        page = WikiHandler(parameters, url = self._url)
        elelist = page.dom.getElementsByTagName('rev')
        if elelist.length is not 0:
            ele = elelist[0]
            s = filter_wiki(ele.childNodes[0].data).encode('sjis', 'ignore')
            result = re.sub(r'[^a-zA-Z ]', '', s.decode('sjis', 'ignore')).lower()
        return result
