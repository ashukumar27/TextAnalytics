# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 09:31:43 2017

@author: Ashutosh
"""

from nltk.book import *


print(type(text2))

##Concordance - words appearing before and after
#Where the word is appearing and in what context
print(text1.concordance('man'))

## Similar - searched the input word and fetches the context of the words, and returns
# all the word in the same context
print(text1.similar('woman'))


#Dispersion PLot - Helpful in analysis of data - see how many times a word has come,
# When did it appear - start or end or middle
# Returns a graph Y(input word) ; XAis( # words, where the word occcur)
# LExical Dispersion PLot
text4.dispersion_plot(['democracy','freedom','law'])



## Counting words in a Text
print(len(text5))

print(text4.count('freedom'))#Count specific word

################################################################################
##########   Frequency Distribution with NLTK    ################################
################################################################################


freqDist = FreqDist(text1)
print(freqDist)
#<FreqDist with 19317 samples and 260819 outcomes>


#words in text
words = freqDist.keys()
print(len(words))
#19317

print(words)
##Outputs all the words in text words

#As a list
words = list(words)
print(words[2:4])

#
print(freqDist['short'])
#59 : Count od a specific word in the text

### PLot
freqDist.plot(20)#MOst common 20 words

#### Usiong your own Text 

text ='Hello! This is a course designed for people who are interested in learning the core concepts of NLP and '\
'utilizing those concepts to make applications to perform sentiment analysis'

print(text)

#FreqDist - input list
text_list = text.split(' ')
print(text_list)

freqDist = FreqDist(text_list)
words = list(freqDist.keys())

print(words)

print(freqDist['of'])


################################################################################
##########                Corpora              ################################
################################################################################

## Accessing Corpora
#nltk corpora
#Gutenburg Corpus
#Web and chat text
#NPS Chat

from nltk.corpus import gutenberg as gt

print(gt.fileids())

#### FUnctions

#Words
bible_kjv = gt.words('bible-kjv.txt')
print(bible_kjv)
#raw - Without doing any linguistic processing
bible_kjv = gt.raw('bible-kjv.txt')
print(bible_kjv)


## For all files in the corpora
for fileid in gt.fileids():
    raw_data = gt.raw(fileid)
    num_words = gt.words(fileid)
    num_sents = gt.sents(fileid)
    num_wrd = len(num_words)
    num_sen = len(num_sents)
    print(fileid, num_wrd, num_sen)
    
    
### Loading your own corpus
from nltk.corpus import PlaintextCorpusReader
import os

#2 inpits - corpus root (path), file ids (regex)



corpus_root = 'D:\DeepLearning\TextAnalytics\Corpora\ShakespearePlays'

print(corpus_root)

file_ids = '.*.txt'
corpus = PlaintextCorpusReader(corpus_root, file_ids)

print(corpus.fileids())

#Words, Sents, len - similar
print(len(corpus.words('shakespeare-hamlet-25.txt')))

################################################################################
##########      Conditional Frequency Distribution        ################################
################################################################################

#Brown Corpus - with categories

from nltk.corpus import brown

#Categories
print(brown.categories())

#Word in categories
print(brown.words(categories='lore'))
print(brown.raw(categories='lore'))
print(brown.sents(categories='lore'))


##Conditional Freq Dist
#Condition - Category

from nltk import ConditionalFreqDist
#Expects a pair of values
# Pir - tuples - first value should be coindition, second word
pair_list = [(category,word) for category in brown.categories() for word in brown.words(categories=category)]

print(pair_list[:10])
print(pair_list[-10:])

freqDist = ConditionalFreqDist(pair_list)
print(freqDist)

print(freqDist['lore']['the'])

print(freqDist.conditions())

#Tabulate function
# Search the word 'the' and 'and' and how many times it comes in each category

category = ['adventure','lore','news']
samples= ['the','and','man']

freqDist.tabulate(conditions=category, samples = samples)


################################################################################
##########      Lexical Resource Vocabulary ################################
################################################################################

#Lexical Resources:
#    Tokenization
#    Homonyms - two similar words (site, cite, sight) ; pike and pike used in different context
#    Stopwords


#input - list of words
#output - list of unusual words

import nltk
def unusual_words(text):
    #normalize text
    text_vocab = set([w.lower() for w in text if w.isalpha()])
    english_vocab = set([w.lower() for w in nltk.corpus.words.words()])
    unusual_list = text_vocab.difference(english_vocab)
    return sorted(unusual_list)

list_unusual_words = unusual_words(gt.words('austen-emma.txt'))

print(list_unusual_words)


#Stopwords corpus
from nltk.corpus import stopwords

print(stopwords.words('english'))













