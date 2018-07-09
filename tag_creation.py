# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:00:53 2018

@author: zakis
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Query.csv', encoding='latin-1')

#Import des documents cleanés
BOW = pd.read_csv('BOW.csv')
BOW = BOW.drop('Unnamed: 0', 1)

tags = [t[1:len(t) - 1].split('><') for t in df['Tags']]

words = []
for t in tags:
     words += t

from collections import Counter
freq_totale = Counter(words)
mostcommon = np.array(freq_totale.most_common(50))

#removing the common words from tags
final_tags = []
for ts in tags:
    final_tags.append([word for word in ts if word in mostcommon])
final_tags = np.array(final_tags)

#removing the empty lists of tags  
# argument where the list of tags is not empty
arg_is_empty = np.array([i for i in range(0, len(final_tags)) if not final_tags[i]]).astype(int)
#array of boolean to keep non empty values
is_not_empty = ~np.in1d(np.arange(0, len(final_tags)), arg_is_empty)
final_tags = final_tags[is_not_empty]
BOW = BOW[is_not_empty]        

length = len(sorted(final_tags,key=len, reverse=True)[0])
y=np.array([xi+[None]*(length-len(xi)) for xi in final_tags])
x = pd.DataFrame(y)
x.to_csv('tags_one_column.csv')

cleaned = x.stack().reset_index(level=1, drop=True).to_frame(''); 
tags = pd.get_dummies(cleaned).groupby(level=0).sum()

tags.to_csv('cleaned_tags.csv')
BOW.to_csv('BOW_supervised.csv')


