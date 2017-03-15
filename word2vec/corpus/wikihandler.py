# coding: utf-8

import urllib
import urllib.parse
import urllib.request
# import urllib2
from xml.dom.minidom import parse as parseXML

from gensim.corpora import WikiCorpus

from gensim.corpora.wikicorpus import filter_wiki

import re

URL = 'http://en.wikipedia.org/w/api.php?'
BASIC_PARAMETERS = {'action': 'query',
                    'format': 'xml'}


class WikiHandler(object):
    def __init__(self, parameters, titles=None, url=URL):
        self._url = url if url.endswith('?') else url + '?'

        self._parameters = {}
        self._parameters.update(BASIC_PARAMETERS)
        self._parameters.update(parameters)

        if titles:
            self._parameters['titles'] = titles

        self.rawdata = self._urlfetch(self._parameters)

        if self._parameters['format'] == 'xml':
            self.dom = parseXML(self.rawdata)
            print('DOM ready.')

    def _urlfetch(self, parameters):
        parameters_list = []

        for key, val in parameters.items():
            if isinstance(val, str):
                val = val.encode('utf-8')
            else:
                val = str(val)

            val = urllib.parse.quote(val)
            parameters_list.append('='.join([key, val]))

        url = self._url + '&'.join(parameters_list)

        print('Accessing...\n', url)

        return urllib.request.urlopen(url, timeout=20)

def category_tracing(lists):
    result = []
    temp_lists = lists
    for index, l in enumerate(lists):
        if 'Category:' in l:
            # print(l)
            temp_lists.pop(index)
            parameters = {'list': 'categorymembers',
                          'cmlimit': 500,
                          'cmtitle': l}
            page = WikiHandler(parameters)
            elelist = page.dom.getElementsByTagName('cm')
            for ele in elelist:
                # print(ele.getAttribute('title').encode('sjis', 'ignore').decode('sjis', 'ignore'))
                result.append(ele.getAttribute('title').encode('sjis', 'ignore').decode('sjis', 'ignore'))

    return temp_lists + result

if __name__ == '__main__':
    # f = open('automobile.articles.txt', 'r')
    # lists = f.readline().split(',')
    # parameters = {'list': 'categorymembers',
    #               'cmlimit': 500,
    #               'cmtitle': 'Category:Automobiles'}
    # page = WikiHandler(parameters)
    # elelist = page.dom.getElementsByTagName('cm')
    # for ele in elelist:
    #     lists.append(ele.getAttribute('title').encode('sjis', 'ignore').decode('sjis', 'ignore'))

    # f0 = open('automobile.articles.10.txt', 'w')
    #
    # for i in range(5):
    #     lists = category_tracing(lists)
    #     f0.write(",".join(lists))
    # f0.close()

    f0 = open('automobile.articles.10.txt', 'r')
    lists = f0.readline().split(",")
    f0.close()

    f = open('automobile.10.en.wiki.txt', 'a')
    i = 0
    for l in lists[71906:]:
        if 'Category:' not in l:
            parameters = {'prop': 'revisions',
                          'rvprop': 'content',
                          'titles': l}
            page = WikiHandler(parameters)
            elelist = page.dom.getElementsByTagName('rev')
            if elelist.length is not 0:
                ele = elelist[0]
                s = filter_wiki(ele.childNodes[0].data).encode('sjis', 'ignore')
                result = re.sub(r'[^a-zA-Z ]', '', s.decode('sjis', 'ignore')).lower()
                f.write(result)
        i += 1
        print(i)

    f.close()
