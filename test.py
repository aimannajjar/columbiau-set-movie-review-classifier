#!/usr/bin/env python
# file: index.py
# author: Aiman Najjar (an2434), Columbia University

"""
Usage: python train.py /path/to/data

    This script trains the NB text-classifier based on the training 
    data set provided in the argument. The training algorithm works
    with two classes_postings. First column in  the CSV file is the label 
    and the second one is the text.

"""
import sys
import os
import logging
import datetime
import cPickle
import zlib
import re
import math
import constants
from nltk import *
from nltk.corpus import wordnet as wn
from PorterStemmer import PorterStemmer
from common import *
from algorithm import *

# Prints Usage
def usage():
    print "usage: python test.py model_file data_file predictions_file [validation_mode]"


if __name__ == "__main__":

    args = sys.argv
    validate_mode = False
    if len(args) < 4 or len(args) > 5:
        usage()    
        sys.exit()

    if len(args) >= 5:
        validate_mode = True


    # Load model file
    f = open(args[1], "r")
    zstr = f.read()
    f.close()
    NB = cPickle.loads(zlib.decompress(zstr))            
    f.close()

    # Initialize variables
    classes_postings = NB["classes_postings"] # For each class there is an entry consisting of
                     # array of documents labeled with that class
    vocabulary = NB["vocabulary"] # For each token there is an entry consisting
                        # of number of occuarnces for each class
    condprob = NB["condprob"]
    selected_features = NB["selected_features"] # list for selected features
    prior = NB["prior"] # prior probabilities for each class

    scores = dict()

    output_str = ""

    # Read test data
    correct_guesses = 0
    avg_diff_correct = 0.0
    avg_diff_incorrect = 0.0
    avg_diff = 0.0
    i = 0

    for line in open(args[2],"r"):

        if line.startswith("Column"):
            continue

        if validate_mode is True:
            parsed_line = line.partition(",")
            expected_label = parsed_line[0]
            doc = parsed_line[2].replace('"', "")                
        else:
            doc = line.replace('"', "")

        scores[i] = dict()

        tokens = word_tokenize(doc)

        if constants.TRAIN_ALGORITHM == "MNB":        
            (scores[i], diff) = ApplyMNB(tokens, classes_postings, condprob, prior, vocabulary, selected_features)
        elif constants.TRAIN_ALGORITHM == "BNB":      
            (scores[i], diff) = ApplyBNB(tokens, classes_postings, condprob, prior, vocabulary, selected_features)


        print("Score for 0: %f" % scores[i]["0"])
        print("Score for 1: %f" % scores[i]["1"])

        diff = math.fabs( scores[i]["0"] - scores[i]["1"])

        if constants.TIEBREAK and diff <= constants.TIEBREAK_MC:
            tiebreaker_scores = tiebreaker(doc, positive_words, negative_words, uncertain_words, swn)
            scores[i]["0"] += tiebreaker_scores["0"]
            scores[i]["1"] += tiebreaker_scores["1"]
            


        max_score = None
        label = None
        for c in scores[i]:
            if max_score is None or scores[i][c] > max_score:
                max_score = scores[i][c]
                label = c


        print "Classified Document: %d " % i
        print "    Text: %s" % doc
        print "    Class: %s" % label
        print "    Score: %f" % max_score
        if validate_mode is True:
            avg_diff += math.fabs( scores[i]["0"] - scores[i]["1"])
            if label == expected_label:
                print "    Correct Guess "
                avg_diff_correct += math.fabs( scores[i]["0"] - scores[i]["1"])
                correct_guesses += 1
            else:
                print "    Incorrect Guess "
                avg_diff_incorrect += math.fabs( scores[i]["0"] - scores[i]["1"])                

        output_str += "%s\n" % label
        print "-----------------------"
        print ""

        i = i + 1


    print ""
    
    if validate_mode is True:
        print "Accuracy: %f" % ((correct_guesses / float(i)) * 100)
        print "Avg. Scores Diff: %f" % (avg_diff / float(i))
        print "Avg. Diff for Correct Guesses: %f" % (avg_diff_correct / correct_guesses)
        print "Avg. Diff for Incorrect Guesses: %f" % (avg_diff_incorrect / (i - correct_guesses))


    print "Vocabulary Size: %d" % len(vocabulary)
    print "Saving predictions_file file"
    f = open(args[3], "w")
    f.write(output_str)
    f.close()
    print "Done"
