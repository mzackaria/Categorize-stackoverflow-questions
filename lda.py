# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:56:33 2018

@author: zakis
"""

import pandas as pd
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 

df = pd.read_csv('Query.csv', encoding='latin-1')

title = df[['Title']]
body = df[['Body']]
tags = df[['Tags']]

lemma = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
reviews = []
for i in range(0, len(body)):
    #put in lower case all the corpus
    review = title.iloc[i,0].lower() + body.iloc[i,0].lower()
    #delete all special characters
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    #tokenize all the words
    tekonized = tokenizer.tokenize(review)
    #stemming
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in tekonized if not word is 'p' or 's' or 'li']
    #delete stop words
    review = [word for word in stemmed if not word in stopwords]
    reviews.append(review)
    
from collections import Counter
#create a list of all the words
words = []
for review in reviews:
    for word in review:
        words.append(word)
        
#count all words
freq_totale = Counter(words)
#take the first n most common words and put it in a set
n = 800
mostcommon = list(np.array(freq_totale.most_common(n))[:,0])
sw = set()
sw.update(mostcommon)
     
#reconstruct every review without the most common words
r = []
for review in reviews:
    review = [word for word in review if not word in sw]
    r.append(review)
    
from gensim import corpora, models
dictionary = corpora.Dictionary(r)
corpus = [dictionary.doc2bow(text) for text in r]
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=50, id2word=dictionary, passes=20)