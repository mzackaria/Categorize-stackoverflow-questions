# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:18:08 2018

@author: zakis
"""


import pandas as pd
import re
import pickle

from bs4 import BeautifulSoup 
from nltk.corpus import stopwords    
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer
#reading the dataset of reviews formed by a title, a body and a set of tags
df = pd.read_csv('Query.csv', encoding='latin-1')

title = df[['Title']]
body = df[['Body']]
tags = df[['Tags']]

#defining the function that will be used to create the dictionnary
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()      
    #
    # 4. Stem all the words
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]            
    #
    # 5. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 6. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 

# Get the number of reviews based on the dataframe column size
num_reviews = body.size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    raw_text =  title.iloc[i].values[0] + ' ' +body.iloc[i].values[0]
    clean_train_reviews.append( review_to_words(raw_text) )
    

# Save the clean train review so we don't have to process it again
filehandler = open("clean_train_reviews.pyc", 'wb')
pickle.dump(clean_train_reviews, filehandler)
    
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 400) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()
df = pd.DataFrame(train_data_features, columns = vocab)
df.to_csv('BOW2.csv')

filehandler = open("vectorizer.pyc", 'wb')
pickle.dump(vectorizer, filehandler)
    
