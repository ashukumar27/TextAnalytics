# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:12:06 2017

@author: Ashutosh

Processing Raw inputs with NLTK
"""

## Tokenization
### Sentence and word tokenization

from nltk.tokenize import word_tokenize,sent_tokenize #words, sentence

def read_file(filename):
    with open(filename,'r') as file:
        text = file.read()
    return text

text = read_file('D:\DeepLearning\TextAnalytics\Corpora\ShakespearePlays\shakespeare-hamlet-25.txt')

print(text)

#word tokenize
words = word_tokenize(text)
print(set(words))

print(len(words))
print(len(set(words))) #Set makes the words unique

#sentence tokenizer
sents = sent_tokenize(text)
for sent in sents:
    print(sents)



################################################################################
##########      Reg Ex                          ################################
################################################################################\\

## Search Function

import re

# 2 parameters - patterm, string

print(re.search('^ab','abc'))
    
if re.search('^ab','abc'):
    print("Found")

words_ending_with_ed= [w for w in words if re.search('ed$',w)]

print(set(words_ending_with_ed))

#Range and Closures
# a* : 0 or many times
# a+ : 1 or many times

print(set([w for w in words if re.search('^a+', str(w).lower())]))

##Searching through tokenized text
from nltk.corpus import gutenberg, nps_chat
import nltk

#nltk.text.Text - Expects an object of Text class


moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))

#findall - Text Class (input - regexp)
#find a phrase " a _____ man" - 3 tokens : a, unknown, man

print(moby.findall(r'<a><.*><man>'))

#Find - unknown unknown bro
chat_obj = nltk.Text(nps_chat.words())

print(chat_obj.findall(r'<.*><.*><bro>'))

## Your own text
text ='Hello! This is a course designed for people who are interested in learning the core concepts of NLP and '\
'utilizing those concepts to make applications to perform sentiment analysis'

our_own_text = nltk.Text(nltk.word_tokenize(text))

print(our_own_text.findall(r'<.*ed>'))

########################################################
############### Stemming     ###########################
########################################################

# Porter - Lancaster
from nltk.stem import PorterStemmer, LancasterStemmer

porter = PorterStemmer()
lanca = LancasterStemmer()

tokens = ['lying']
print(porter.stem(tokens[0]))
print(lanca.stem(tokens[0]))


########################################################
############### Lemmatizer   ###########################
########################################################

#only removes affixes from words, plural to singular

from nltk.stem import WordNetLemmatizer
lemma  = WordNetLemmatizer()

tokens = brown.words(categories = ['religion'])

print(set([lemma.lemmatize(t) for t in tokens]))


text = "The women are lying"
tokens = nltk.word_tokenize(text)

print([lemma.lemmatize(t) for t in tokens])


########################################################
############### Regex for Tokenization   ###############
########################################################


text = "Gokhale conveyed that he was in Hong-Kong and could reach only past midnight even if he booked himself on the first Beijing-bound flight. He was urged to reach the Chinese capital as fast as he could, in a first clear indication that the quiet and dogged attempt to defuse the Doklam imbroglio may have borne fruit."

print(re.split(' ',text))

##Removing fullstop in the end

print(re.split('\s',text))  #\s: space, tab, enter

print(re.split('\W',text)) #\W: All non-alphabet characters including hyphen Hong-King
#to get the hypenized words

#print(re.findall('\w+|\S|\w*',text))
print(re.findall("\w+[-']+\w+|\w+",text)) #Returns hyphenated characters


