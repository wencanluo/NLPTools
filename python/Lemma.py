#!/usr/bin/python

#This script is used to get the lemma of a word using WordNet with NLTK toolkit
#Author: Wencan Luo (fixed-term.Wencan.Luo@us.bosch.com)
#Date: 08/01/2013
#
#To use it, you have to install NLTK (http://nltk.org/) first.

import nltk
from nltk.corpus import wordnet as wn 

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

def lemmatizeSentence(sentence):
	"""
	@Function: Lemmatize a sentence. 
	@param sentence: string, a string of words (splity by " ") 
	@Example:
		 "finding friend location"=lemmatizeSentence("finding friends locations")
	"""
	
	global lmtzr
	ls = ""
	for word in sentence.split(): 
		word = lmtzr.lemmatize(word)
		ls = ls + word + " "
	return ls.strip()

def lemmatize(word):
	"""
	@Function: Lemmatize a word. 
	@param word: string, the input word 
	@Example:
		 friend=lemmatize("friends")
	"""
	
	global lmtzr
	lemma = lmtzr.lemmatize(word)
	return lemma
	
if __name__ == "__main__":
	T = "my friends are in palo alto"
	
	Ts = T.split()
	
	for i in range(len(Ts)):
		print lemmatize(Ts[i])
	
	print lemmatize("said")
	print lemmatizeSentence("finding friends locations")
	print "Done!"