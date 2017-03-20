#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:23:23 2016

@author: Hideyoshi Yanagisawa, University of Tokyo
"""

from flask import Flask, render_template, request, redirect, url_for, g
from nltk.corpus import wordnet as wn
#import numpy as np
#from markupsafe import Markup
import pandas as pd
pd.options.display.max_colwidth =200
# User defined 
from MultiLangTrans import suggestWordsFromSynsets
from wn_extension import get_wnaffect_category,get_domain_category,get_senti_values

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
        
    return pd.DataFrame({'synset':s_name,
                        'pos':pos,
                        'meaning':s_def,
                        'select':ch_vec,
                        'domain':domain_categ,
                        'affect':affect_categ,
                        'P':em_p,
                        'N':em_n,
                        'Obj':em_o,
                        },
                        columns=['synset','pos','meaning','domain',
                        'affect','P','N','Obj','select'])

# app instance
app = Flask(__name__)

# index routing
@app.route('/')
def index():
    S_in = getattr(g, 'S_in', None)
     word_in = getattr(g, 'word_in', None)
    lang_in = getattr(g, 'lang_in', None)
    lang_out = getattr(g, 'lang_out', None)
    
    g.S_in = [];g.word_in=""
    g.lang_out="";g.lang_in=""
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
            g.word_in = request.form['word_in']
            g.pos = request.form['pos']
            g.lang_in = request.form['lang_in']
            g.lang_out = request.form['lang_out']
            
            # Get Synset
            if g.pos =="x":
                g.S_in = wn.synsets(g.word_in, lang=g.lang_in)
            else:
                g.S_in = wn.synsets(g.word_in, lang=g.lang_in, pos = g.pos)

            # render synset.html
            S_in_table = make_synset_table(g.S_in)
            synset_html = S_in_table.to_html\
                            (classes='table table-striped" id = "synsettable',escape=False)
            return render_template('synset.html',title='MTS: Synset list',
                                   synset=synset_html,word_in=g.word_in)
        
        elif action == "suggest":
            # get selected synset
            ch = request.form.getlist("ch_synset")
            ch = [int(i) for i in ch]
            g.S_in = [g.S_in[i] for i in ch]
            # search words
            sgwords = suggestWordsFromSynsets(g.S_in,g.lang_out)
            sgwords=sgwords.round(2)
            # render wordlist.html
            sgwords_html = sgwords.to_html\
                            (classes='table table-striped" id = "wordtable',escape=False)

            return render_template('wordlist.html',title='MTS: Suggested words',
                                   sgwords=sgwords_html\
                                   ,word_in=g.word_in,lang_in=g.lang_in,lang_out=g.lang_out)            
    else:
        return redirect(url_for('index')) # Redirect in case of error 

if __name__ == '__main__':
    app.debug = True # debug mode
    #app.run()
    app.run(host='0.0.0.0') # can be accessed from anywhere
