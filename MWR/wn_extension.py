#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:42:50 2016

@author: hidey
"""

import sqlite3

def get_domain_category(synname):
    affectdb = './db/wn-domain.db'
    conn = sqlite3.connect(affectdb)
    sql = "SELECT c1,c2,c3,c4,c5,c6,c7 FROM domain WHERE synset='%s'" % (synname)
    cur = conn.execute(sql)
    categ=''
    for c in cur: 
        for s in c:
            if s!='':categ = categ+s+'-'
        categ = categ[:-1]+' '
    if len(categ)>1:categ = categ[:-1]
    return categ

def get_wnaffect_category(synname,pos):
    affectdb = './db/eng-asynset.db'
    conn = sqlite3.connect(affectdb)
    posdict = {'a':'adjsyn','s':'adjsyn','r':'advsyn'\
                ,'n':'nounsyn','v':'verbsyn'}
    sql = "SELECT categ FROM %s WHERE synset='%s'"\
            % (posdict[pos],synname)
    cur = conn.execute(sql)
    categ=''
    for c in cur: categ = categ+c[0]+', '
    if len(categ)>2:categ = categ[:-2]
    return categ

    
from nltk.corpus import sentiwordnet as swn    
def get_senti_values(syn_name):
    score = swn.senti_synset(syn_name) 
    vec = [score.pos_score(),score.neg_score(),score.obj_score()]
    return vec

    