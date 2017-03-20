#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:12:29 2016

@author: Hideyoshi Yanagisawa, University of Tokyo
"""

from nltk.corpus import wordnet as wn
from collections import Counter
import pandas as pd

# User defined
from commonality import Commonality
    
def suggestWordsFromSynsets(S_in,lang_out):
    words = []
    for s_in in S_in:
        words.extend(s_in.lemmas(lang_out))
    # Get unique of wordvec and counts
    W_out = []; contvec = []
    counter = Counter(words)
    for word,cnt in counter.most_common():
        W_out.append(word.name())
        contvec.append(cnt)

    S_in = set(S_in)
    sim=[];covg=[];extr=[];univ=[];syn_sh=[];syn_ex=[];syn_ms=[]    
    n_syn = []
    for w in W_out:
        # get S_out
        S_out = set(wn.synsets(w,lang=lang_out))
        #print(S_out)
        
        # Calc
        S_and = S_in & S_out
        S_or = S_in | S_out
        S_risk = S_out - S_in
        S_miss = S_in - S_out
        consistency = len(S_and) / len(S_or)  
        coverage = len(S_and) / len(S_in)
        risk =  1.0 - len(S_and)/len(S_out)
        
        n_syn.append(len(S_out))
        sim.append(consistency)
        covg.append(coverage)
        extr.append(risk)
        univ.append(Commonality(w,lang_out))
        syn_sh.append([s.definition() for s in S_and])
        syn_ex.append([s.definition() for s in S_risk])
        syn_ms.append([s.definition() for s in S_miss])        
#        syn_sh.append([s.name()+': '+s.definition() for s in S_and])
#        syn_ex.append([s.name()+': '+s.definition() for s in S_risk])
#        syn_ms.append([s.name()+': '+s.definition() for s in S_miss])           
    tbl = {'words':W_out
               ,'similarity':sim\
               ,'# meaning':n_syn\
               ,'universal':univ\
               ,'shared meaning':syn_sh\
#               ,'coverage':covg\
#               ,'extra':extr\
#               ,'extra meaning':syn_ex\
#               ,'missing meaning':syn_ms\
               }
#    col=['words', 'similarity', 'coverage','extra','universal',
#             'shared synset','extra synset','missing synset']
    col=['words', '# meaning', 'similarity', 'universal','shared meaning']
               
    #metrics.update({'words':W_out})    # Output table
    
    res = pd.DataFrame(tbl,columns=col)
    #res = pd.DataFrame(tbl)
    return res.sort_values(by=["similarity","universal"], ascending=[False,True])
    
def suggestWord(word_in,lang_in,lang_out):
    # Get synsets of input word
    S_in = wn.synsets(word_in, lang=lang_in)
    if len(S_in)==0:
        res = []
    else:
        for s_in in S_in:
            print(s_in.name()+": "+s_in.definition())
        res = suggestWordsFromSynsets(S_in[0:1],lang_out)
    # synset selection
    return res
    
if __name__ == '__main__':
    word_in = "beautiful"
    lang_in = "eng"
    lang_out = "jpn"
    
    res=suggestWord(word_in,lang_in,lang_out)
    print(res)
    
    #print(res.sort_values(by=["similarity"], ascending=False))

 
    '''            
    # Synset combunation to calculate similarities
    S_in_v = [];S_out_v = [];
    wi = 0; w_out_idx = []
    for w in W_out:
        st = wi
        S_out = wn.synsets(w,lang=lang_out)
        for s_in in S_in:
            for s_out in S_out:
                S_in_v.append(syname_NLTK2Perl(s_in.name()))
                S_out_v.append(syname_NLTK2Perl(s_out.name()))
                wi+=1
        ed = wi-1
        w_out_idx.append([st,ed])
    
    # Calculate similarities and their averaging 
    S_in_csv = list2csv(S_in_v)
    S_out_csv = list2csv(S_out_v)
    sim_vec = perl_similarities(S_in_csv,S_out_csv) # Calculate similarities
    # Average similarity for each word
    avg_sim_lst = []
    for idx in w_out_idx:
        avg = sum(sim_vec[idx[0]:idx[1]]) / (idx[1]-idx[0]+1)
        avg_sim_lst.append(avg)
    '''

'''
import subprocess
def perl_similarities(syns1,syns2):
    #set1 = 'aggressive#a#1,aggressive#a#2'
    #set2 = 'sportive#a#1,sportive#a#2'
    simvec = []
    cmd = ['perl', './glossvec.pl', syns1, syns2]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in proc.stdout:
        simvec.append(float(line))
    return simvec
    
def syname_NLTK2Perl(sn):
    sn = sn.replace('.s.','#a#')
    sn = sn.replace('.','#')
    return sn

def list2csv(lst):
    lst = map(str,lst)
    return ",".join(lst)    

def calc_sim(S_in,S_out):
    S_and = set(S_in)&set(S_out)
    S_or = set(S_in)|set(S_out)
    sim = len(S_and) / len(S_or)      
    return sim    
'''