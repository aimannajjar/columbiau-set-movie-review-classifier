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
import constants
import math
import array
import numpy
import thread
from threading import Thread
from PorterStemmer import PorterStemmer
from BitVector import BitVector
from nltk import *
from common import *
from algorithm import *

# Prints Usage
def usage():
    print "usage: python train.py /path/to/dataset_directory output_file [max_docs]"

def file_len(fname):
    """Returns total number of lines in file fname"""
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def initVector(no, ref, start, count, vec_size):
    for i in range(start, start+count):
        ref[i] = BitVector(intVal =0, size=vec_size)
        print "Thread-%d init'ed %d" % (no, i)


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    args = sys.argv
    if len(args) < 3 or len(args) > 4:
        usage()    
        sys.exit()

    MAX_DOCS = 0
    if len(args) >= 4:
        MAX_DOCS = int(args[3])

    # --------------------------------------------------------------------------
    # Initialize variables
    classes_postings = dict() # For each class there is an entry consisting of
                     # array of documents labeled with that class
    docs = dict()
    vocabulary = dict() # For each token there is an entry consisting
                        # of number of occuarnces for each class

    vocabulary_postings = dict() # For each token, list of documents that contain
                                 # this term per class

    T = dict() # bit vector for each term
    C = dict() # bit vector for each class


    text = dict() # For each class there is an entry consisting of
                  # the concatenated text of all documents in the classs
    total_occurances = dict() # For each class, there is an entry which
                              # is a count of total terms occurances




    # Load Dictionaries
    stop_words = load_dictionary("stop_words.txt")

    # Calculate total number of documents
    N = 0
    for filename in os.listdir(args[1]):
        if not filename.endswith(".csv"):
            continue
        N += file_len("%s/%s" % (args[1], filename))
        if MAX_DOCS != 0 and N >= MAX_DOCS:
            N = MAX_DOCS
            break

    MAX_DOCS = N


    # Initialize vectors (only needed for MI algorithm)
    if constants.FEATURE_SELECTION is not None:
        print "Initializing Vectors"
        a = datetime.datetime.now()
        
        max_vectors = 50000
        per_thread = int(constants.K_FEATURES / constants.NUM_THREADS)
        VectorPool = [None] * max_vectors
        threads = []
        for f in range(0, int(len(VectorPool) / per_thread)):
            print "Starting thread: %d" % f
            th = Thread(target=initVector, args=(f, VectorPool, f * per_thread, per_thread, N,))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()
            print "Thread %s finished" % th.getName()
        b = datetime.datetime.now()
        print "%d Vectors initialized in %0.4f seconds " % (max_vectors, ( (b-a).microseconds / 1000.0) / 1000.0 )

    print "Total Documents: %d" % N


    # -------------------------------
    # Feature Extraction
    # -------------------------------
    i = 0
    pool_i = 0
    for filename in os.listdir(args[1]):
        if not filename.endswith(".csv"):
            continue

        # Each line represents a document
        for line in open("%s/%s" % (args[1], filename)):

            if line.startswith("Column"): # Skip header row
                continue

            print "Processing Document [dataset: %s]: %d / %d" % (filename, i+1, N)

            parsed_line = line.partition(",")
            label = parsed_line[0]
            doc = parsed_line[2].replace('"', "")

            docs[i] = doc

            if label not in classes_postings:
                classes_postings[label] = []
                text[label] = ""
                total_occurances[label] = 0
                if constants.FEATURE_SELECTION is not None:
                    C[label] = BitVector(intVal =0, size=N)

            if constants.FEATURE_SELECTION is not None:
                C[label][i] = 1

            classes_postings[label].append(i)
            text[label] = text[label] + " " + doc

            tokens = word_tokenize(doc)
            for token in tokens:
                if token in stop_words:
                    continue

                if len(token) <= 2:
                    continue 

                token = process_word(token)

                if token not in vocabulary:
                    vocabulary[token] = dict()
                    vocabulary_postings[token] = dict()
                    if constants.FEATURE_SELECTION is not None:
                        T[token] = VectorPool[pool_i]
                        pool_i += 1

                if label not in vocabulary[token]:
                    vocabulary[token][label] = 0
                    vocabulary_postings[token][label] = set([])

                vocabulary[token][label] = vocabulary[token][label] + 1
                vocabulary_postings[token][label].add(i)
                total_occurances[label] = total_occurances[label] + 1  
                if constants.FEATURE_SELECTION is not None:          
                    T[token][i] = 1

            i = i + 1

            if i >= MAX_DOCS:
                break

        if i >= MAX_DOCS:
            break



    # -------------------------------
    # Feature Selection
    # -------------------------------
    selected_features = dict()

    if constants.FEATURE_SELECTION is not None:
        # Compute Mutual Information
        print("Computing Mutual Information values")
        A = compute_mi(vocabulary,classes_postings, T, C, N)
        print("Done")

        # Select features
        target_k = constants.K_FEATURES
        target_atc = 0.9
        for c in classes_postings:
            selected_features[c] = dict()
            k = 0
            for t in sorted(A, key=lambda t: A[t][c], reverse=True):
                if k < target_k:
                    print "A(%s,%s) = %f" % (t,c, A[t][c])
                    selected_features[c][t] = True
                elif c in vocabulary[t]:
                    total_occurances[c] -= vocabulary[t][c]
                    vocabulary[t][c] = 0

                    print "- A(%s,%s) = %f" % (t,c, A[t][c])

                k = k+1


    # -------------------------------
    # Training Algorithm
    # -------------------------------
    print "Training"
    if constants.TRAIN_ALGORITHM == "MNB":
        print "Algorithm: %s" % constants.TRAIN_ALGORITHM
        (condprob, prior) = TrainMNB(classes_postings, vocabulary, total_occurances, N)
    elif constants.TRAIN_ALGORITHM == "BNB":
        print "Algorithm: %s" % constants.TRAIN_ALGORITHM
        (condprob, prior) = TrainMNB(classes_postings, vocabulary, total_occurances, N)



    # -------------------------------
    # Save & Exist
    # -------------------------------
    # Dump model file on disk
    NB = dict()
    NB["prior"] = prior
    NB["condprob"] = condprob
    NB["vocabulary"] = vocabulary
    NB["classes_postings"] = classes_postings
    NB["selected_features"] = selected_features

    f = open(args[2], "w")
    f.write(zlib.compress(cPickle.dumps(NB,cPickle.HIGHEST_PROTOCOL),9))
    f.close()




