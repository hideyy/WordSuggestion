#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:12:29 2016

@author: hidey
"""
from nltk.corpus import wordnet as wn
from collections import Counter
    

def calc_sim(S_in,S_out):
    S_and = set(S_in)&set(S_out)
    S_or = set(S_in)|set(S_out)
    sim = len(S_and) / len(S_or)      
    return sim        
    
'''
def calc_sim2(S_in,S_out):  
    S_and = S_in & S_out
    S_or = S_in | S_out
    S_risk = S_out - S_in
    S_miss = S_in - S_out
    consistency = len(S_and) / len(S_or)  
    coverage = len(S_and) / len(S_in)
    risk =  1.0 - len(S_and)/len(S_out)
'''

def max_no(x):
    return max(enumerate(x), key=lambda x: x[1])[0]

def Commonality(word_in,lang_in):    
    langs = {'eng','cmn','jpn','fra','spa'}    
    S_in = wn.synsets(word_in, lang=lang_in)
    if len(S_in)==0:
        comm = 0
    else:    
        lang_set = langs - {lang_in}
        # print(lang_set)
        comm = 0
        for l in lang_set:
            words = []
            for s_in in S_in:
                if len(s_in.lemmas(l))!=0: 
                    words.extend(s_in.lemmas(l))
            # Get unique of wordvec and counts
            W_out = []; contvec = []
            
            if len(words)!=0:
                counter = Counter(words)
                for word,cnt in counter.most_common():
                    W_out.append(word.name())
                    contvec.append(cnt)
                #print(counter)
                sim_lst = []
                for w in W_out:
                    S_out = wn.synsets(w,lang=l)
                    sim_lst.append(calc_sim(S_in,S_out))
                sim_max = max(sim_lst)
                #w_max = W_out[max_no(sim_lst)]
                #print('%s: %f' % (w_max,sim_max,))
                comm += sim_max
        comm = comm / len(lang_set)   
    return comm    
 
if __name__ == '__main__':
    c = Commonality('good','eng')
    print(c)