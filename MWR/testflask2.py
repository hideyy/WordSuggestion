#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:40:50 2016

@author: hidey
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello World!"

if __name__ == '__main__':
    app.run()
    