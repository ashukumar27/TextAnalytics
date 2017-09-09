# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:50:15 2017

@author: Ashutosh

Categorization & Tagging words
"""

########################################################
###############      Tagger      ###############
########################################################

import nltk

text = 'We are learning Natural language Processing'

tokens = nltk.word_tokenize(text)

print(tokens)

#POS : Part of Speech Tagging
#pos_tag() - list of tokens as input

print(nltk.pos_tag(tokens))

print(nltk.help.upenn_tagset('VBP'))

#similar method of text
tokens = [word.lower() for word in nltk.corpus.brown.words()]

text = nltk.Text(tokens)

print(text.similar('woman')) # goes thru the text and find ths contxt as part of seppech context

print(nltk.pos_tag(['woman']))



########################################################
###############      Tagged Corpus      ###############
########################################################
sample = "My name is Ashutosh. I work at Seynse"

#Find similar context words as Ashutosh - names
##Create a text object
##Call similar method - maps the context of searched words using POS tag, and finds all

#Perform tagging once, and save

word = 'Australia/NNP'

#String to tuple : str2tuple
print(nltk.tag.str2tuple(word))
#we can save tags to save processing 

#Using Brown Corpus - tagged words
#Brown corpus is saved with POS tags
for word,tag in nltk.corpus.brown.tagged_words(categories="news"):
    print(word," " ,tag)


########################################################
###############    The Default Tagger        ###############
########################################################
from nltk.corpus import brown
import nltk

tags = [tag for (word,tag) in brown.tagged_words(categories='news')]

print(set(tags))

#Finding tags occuring most frew
print(nltk.FreqDist(tags).max())

## default Tagger - tags as input, marks every word in the tagger as given tag

default_tagger = nltk.DefaultTagger('NN')
default_tagger = nltk.DefaultTagger(nltk.FreqDist(tags).max())

text = "Gokhale conveyed that he was in Hong-Kong and could reach only past midnight even if he booked himself on the first Beijing-bound flight. He was urged to reach the Chinese capital as fast as he could, in a first clear indication that the quiet and dogged attempt to defuse the Doklam imbroglio may have borne fruit."
tokens = nltk.word_tokenize(text)

print(default_tagger.tag(tokens))

##Performace of a tagger
print(default_tagger.evaluate(nltk.corpus.brown.tagged_sents(categories='news')))

########################################################
###############     Regexp Tagger       ###############
########################################################
import nltk
pattern = [(r'.*ing$', 'VBG'),(r'.*','NN')] #tuples

sample = "I am playing football"

regexp_tagger =  nltk.RegexpTagger(pattern)

print(regexp_tagger.tag(nltk.word_tokenize(sample)))

########################################################
###############    Unigram Tagger        ###############
########################################################

#statistically analyzed the word and figures out how it is used more frequently -
# in our sentence it might appear as an adjective but in general it is used as a verb

#I frequently visit this bakery. This is a frequent word : 
    #first frequent is verb, second frequent is adjective
    
from nltk.corpus import brown
import nltk

#get tagged sentences
# to train unigram tagger
brown_tagged_sents = brown.tagged_sents(categories='news')

#get untaggged sents
#to test tagger
brown_sents = brown.sents(categories = 'news')

unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)

print(unigram_tagger.tag(nltk.word_tokenize('I am studying NLP')))
#None for unseen word

print(unigram_tagger.evaluate(brown_tagged_sents))

#Separating Training and Testing

size = int(len(brown_tagged_sents)*0.9)

train = brown_tagged_sents[:size]
test = brown_tagged_sents[size:]

unigram_tagger = nltk.UnigramTagger(train)

print(unigram_tagger.evaluate(test))

########################################################
###############    NGram Tagger        ###############
########################################################


#Judges the tag based on the other N-1 tags, analyzes word and context
brown_tagged_sents = brown.tagged_sents(categories='news')

brown_sents = brown.sents(categories = 'news')

#Ngram tagger - expects a value of N - num of tokes to judge the tag 
ngram_tagger = nltk.NgramTagger(4, train= brown_tagged_sents)


print(ngram_tagger.tag(nltk.word_tokenize('We are studying NLP')))


