#!/usr/bin/python
import sys
import math
import constants
from nltk.corpus import wordnet as wn
from sentiwordnet import SentiWordNetCorpusReader, SentiSynset
from nltk import *
from common import *

# Load global dictionaries
swn_filename = 'dictionaries/SentiWordNet_3.0.0_20120510.txt'
swn = SentiWordNetCorpusReader(swn_filename)
positive_words = load_dictionary("positive_words.txt", True)
negative_words = load_dictionary("negative_words.txt", True)
uncertain_words = load_dictionary("uncertain_words.txt", True)
stop_words = load_dictionary("stop_words.txt", False)
negation_words = load_dictionary("negations.txt", False)
names = load_dictionary("names.txt", False)



def TrainMNB(classes, vocabulary, total_occurances, N):
    condprob = dict()
    prior = dict()

    for c in classes:
        Nc = len(classes[c])
        prior[c] = float(Nc) / float(N)

        for t in vocabulary:

            if c not in vocabulary[t]:
                vocabulary[t][c] = 0

            Tct = vocabulary[t][c]

            if t not in condprob:
                condprob[t] = dict()

            if c not in condprob[t]:
                condprob[t][c] = 0.0

            condprob[t][c] = float(Tct + 1) / (total_occurances[c] + 1)

    return (condprob, prior)


def ApplyMNB(doc_tokens, classes_postings, condprob, prior, vocabulary, selected_features):
    ## Assumes global dictionaries defined: stop_words, names, negation_words
    global stop_words, names, negation_words
    scores = dict()
    for c in classes_postings:
        scores[c] = 0 # math.log(prior[c])                        

        negation_found = False
        adverb_found = False
        adverb_condprob = 0.0
        for t in doc_tokens:
            t = t.lower()

            if constants.LA and t in negation_words:
                negation_found = True
                continue

            if t in stop_words:
                continue

            if t in names:
                continue

            
            isAdj = wn.morphy(t, wn.ADJ) is not None
            isNoun = wn.morphy(t, wn.NOUN) is not None
            isVerb = wn.morphy(t, wn.VERB) is not None
            isAdv = wn.morphy(t, wn.ADV) is not None


            if constants.LA and negation_found:
                negation_found = False
                continue


            t = process_word(t)

            if t not in vocabulary:
                continue
            if constants.FEATURE_SELECTION is not None and t not in selected_features[c]:
                continue


            condp = condprob[t][c]
            scores[c] = scores[c] + math.log(condp)
    diff = math.fabs(scores["0"] - scores["1"])

    return (scores, diff)



def TrainBNB(classes, vocabulary_postings, total_occurances, N):
    condprob = dict()
    prior = dict()

    for c in classes:
        Nc = len(classes[c])
        prior[c] = float(Nc) / float(N)

        for t in vocabulary_postings:

            if c not in vocabulary_postings[t]:
                vocabulary_postings[t][c] = set([])

            Nct = len(vocabulary_postings[t][c])

            if t not in condprob:
                condprob[t] = dict()

            if c not in condprob[t]:
                condprob[t][c] = 0.0

            condprob[t][c] = float(Nct + 1) / (Nc + 2)

    return (condprob, prior)



def ApplyBNB(doc_tokens, classes_postings, condprob, prior, vocabulary, selected_features):
    ## Assumes global dictionaries defined: stop_words, names, negation_words
    global stop_words, names, negation_words
    scores = dict()
    for c in classes_postings:
        scores[c] = 0 #math.log(prior[c])                        

        negation_found = False
        adverb_found = False
        adverb_condprob = 0.0
        doc_features = []
        for t in doc_tokens:
            t = t.lower()

            if constants.LA and t in negation_words:
                negation_found = True
                continue

            if t in stop_words:
                continue

            if t in names:
                continue

            
            isAdj = wn.morphy(t, wn.ADJ) is not None
            isNoun = wn.morphy(t, wn.NOUN) is not None
            isVerb = wn.morphy(t, wn.VERB) is not None
            isAdv = wn.morphy(t, wn.ADV) is not None


            if constants.LA and negation_found:
                negation_found = False
                continue


            t = process_word(t)

            if t not in vocabulary:
                continue
            if constants.FEATURE_SELECTION is not None and t not in selected_features[c]:
                continue

            doc_features.append(t)

        vocab = vocabulary
        if constants.FEATURE_SELECTION is not None:
            vocab = selected_features[c]

        for t in vocabulary:
            if t in doc_features:
                scores[c] += math.log(condprob[t][c])
            else:
                scores[c] += math.log(1 - condprob[t][c])


    diff = math.fabs(scores["0"] - scores["1"])

    return (scores, diff)




def compute_mi(vocabulary,classes_postings,T,C, N):
    MI = dict() # Mutual information values -- i.e. A(t,c)     
    print("Vocabulary Size: %d" % len(vocabulary))
    z = 1
    Nv = len(vocabulary)
    for t in vocabulary:
        MI[t] = dict()
        print("Processing term %d / %d" % (z, Nv))

        for c in classes_postings:
            MI[t][c] = 0.0
            # -----------------------------
            # Let's compute MI factors
            # N00: Docs don't contain the term and not in the class            
            # N01: Docs don't contain the term but in the class            
            # N10: Docs contain the term but not in the class            
            # N11: Docs contain the term and in the class

            NT = ~T[t]
            NC = ~C[c]
            N11 = (T[t] & C[c]).count_bits() + 1.0
            N00 = (NT & NC).count_bits() + 1.0
            N10 = (T[t] & NC).count_bits() + 1.0
            N01 = ( NT & C[c]).count_bits() + 1.0

            N1x = N11 + N10 
            Nx1 = N11 + N01 
            Nx0 = N00 + N10 
            N0x = N00 + N10 

            MI[t][c] =  ((N11 / N) * math.log((N * N11) / (N1x * Nx1), 2))  
            MI[t][c] += ((N01 / N) * math.log((N * N01) / (N0x * Nx1), 2))  
            MI[t][c] += ((N10 / N) * math.log((N * N10) / (N1x * Nx0), 2))   
            MI[t][c] += ((N00 / N) * math.log((N * N00) / (N0x * Nx0), 2))



        z += 1    

    return MI



def tiebreaker(doc, positive_list, negative_list, uncertain_list, swn):
    tokens = word_tokenize(doc)
    positive_words = 0
    negative_words = 0
    uncertain_words = 1
    scores = { "0": 0.0 , "1": 0.0}

    for token in tokens:
        token = token.lower().strip()
        t = process_word(token)

        if stem in positive_list:
            positive_words += 1
        elif stem in negative_list:
            negative_words += 1
        elif stem in uncertain_list:
            uncertain_words += 1

        if constants.TIEBREAK_WN:
            try:
                word_syn = swn.senti_synsets(token)[0]
                scores["0"] += (word_syn.neg_score * 2)
                scores["1"] += (word_syn.pos_score * 2)
            except Exception, e:
                pass

    if constants.TIEBREAK_DT:
        scores["0"] += ((negative_words - positive_words) *  (1.0 /  uncertain_words))
        scores["1"] += ((positive_words - negative_words) *  (1.0 /  uncertain_words))

    return scores

