#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:23:23 2016

@author: hidey
"""

from flask import Flask, render_template, request, redirect, url_for
from nltk.corpus import wordnet as wn
#import numpy as np
#from markupsafe import Markup
import pandas as pd
pd.options.display.max_colwidth =200
# User defined 
from MultiLangTrans import suggestWordsFromSynsets
from wn_extension import get_wnaffect_category,get_domain_category,get_senti_values

# GLOBAL variables #############
GLB = {}
S_in = []       # Synset list
word_in = ""
lang_in = ""
lang_out = ""
################################

# Make synset table
def make_synset_table(S):
    s_name =[]; s_def = []; pos = []
    ch_vec = []
    affect_categ = []
    domain_categ = []
    em_p = [];em_n = [];em_o=[]
    i = 0
    for s in S:
        s_name.append(s.name())
        s_def.append(s.definition())
        pos.append(s.pos())
        ch = '<div class="checkbox"><label>\
        <input type="checkbox" name="ch_synset" value="%d" checked="checked">\
        </label></div>' % (i,)
        ch_vec.append(ch)
        i+=1
        # wn-affect
        afcat = get_wnaffect_category(s.name(),s.pos())
        affect_categ.append(afcat)
        # wn-domain
        dmcat = get_domain_category(s.name())
        domain_categ.append(dmcat)
        # Senti-WN
        emvec = get_senti_values(s.name())
        em_p.append(str(emvec[0]))
        em_n.append(str(emvec[1]))
        em_o.append(str(emvec[2]))
        
    return pd.DataFrame({'ID':s_name,
                        'meaning':s_def,
                        'select':ch_vec,
                        'positive':em_p,
                        'negative':em_n,
                        'objective':em_o,
                        },
                        columns=['ID','meaning',
                        'positive','negative','objective','select'])
#    return pd.DataFrame({'synset':s_name,
#                        'pos':pos,
#                        'meaning':s_def,
#                        'select':ch_vec,
#                        'domain':domain_categ,
#                        'affect':affect_categ,
#                        'P':em_p,
#                        'N':em_n,
#                        'Obj':em_o,
#                        },
#                        columns=['synset','pos','meaning','domain',
#                        'affect','P','N','Obj','select'])
# app instance
app = Flask(__name__)

# index routing
@app.route('/')
def index():
    global GLB
    GLB = {"S_in":[],"word_in":"","pos":"","lang_in":"","lang_out":""}
    return render_template('index.html',title='MTS')

# /post routing 
@app.route('/post', methods=['GET', 'POST'])
def post():
    global GLB
    if request.method == 'POST':
        action = request.form['action']
        # Show Synsets
        if action == "synset":
            # Get word and attributes from text form
            word_in = request.form['word_in']
            pos = request.form['pos']
            lang_in = request.form['lang_in']
            lang_out = request.form['lang_out']
            
            # Get Synset
            if pos =="x":
                S_in = wn.synsets(word_in, lang=lang_in)
                #print(S_in)
            else:
                S_in = wn.synsets(word_in, lang=lang_in, pos = pos)
            
            # renew GLB
            GLB = {"S_in":S_in,"word_in":word_in,"pos":pos\
                   ,"lang_in":lang_in,"lang_out":lang_out}

            # render synset.html
            S_in_table = make_synset_table(S_in)
            synset_html = S_in_table.to_html\
                            (classes='table table-striped" id = "synsettable',escape=False)
            return render_template('synset.html',title='MTS: Synset list',
                                   synset=synset_html,word_in=GLB['word_in'])
        # Suggest words
        elif action == "suggest":
            # get selected synset
            ch = request.form.getlist("ch_synset")
            ch = [int(i) for i in ch]
            S_in = [GLB['S_in'][i] for i in ch]
            # search words
            sgwords = suggestWordsFromSynsets(S_in,GLB['lang_out'])
            sgwords=sgwords.round(2)
            #print(sgwords)
            # render wordlist.html
            sgwords_html = sgwords.to_html\
                            (classes='table table-striped" id = "wordtable',escape=False)

            
            return render_template('wordlist.html',title='MTS: Suggested words',
                                   sgwords=sgwords_html\
                                   ,word_in=GLB['word_in'],lang_in=GLB['lang_in'],lang_out=GLB['lang_out'])            
    else:
        # Redirect in case of error 
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # debug mode
    #app.run()
    app.run(host='0.0.0.0') # どこからでもアクセス可能に
