#!/usr/bin/env python

import nltk
import nltk.data

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(paragraph)
    

if __name__ == '__main__':
    print splitSentence("[1] I love you. [2] Sent 2. [3] sentence 3")