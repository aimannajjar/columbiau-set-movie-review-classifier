#!/usr/bin/python

from PorterStemmer import PorterStemmer
import constants

def load_dictionary(filename, stem=True):
    """Loads line separated dictionary into a list"""
    out = []
    for word in open("dictionaries/%s" % filename, "r"):
        word = word.lower()
        if stem is True:
            p = PorterStemmer()
            word = p.stem(word, 0,len(word)-1)               
        out.append(word)
    return out


def process_word(token):
    token = token.lower()
    if constants.STEM is True:
        p = PorterStemmer()
        token = p.stem(token, 0,len(token)-1)                       
    
    return token
