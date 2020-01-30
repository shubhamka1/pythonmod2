#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle 


# In[ ]:


news= input("enter news:")


# In[ ]:


with open('randomforestmodel.sav', 'rb') as file:
    pickle_model = pickle.load(file)


# In[ ]:


count_vectorizer = CountVectorizer()
input_vector = count_vectorizer.fit_transform(news)


# In[ ]:


pred= pickle_model.predict(news)


# In[ ]:


print(input_vector)


# In[ ]:




