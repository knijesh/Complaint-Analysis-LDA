#!/usr/bin/env python
# coding: utf-8

# # Load and install necessary packages
# 

# In[64]:


import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# In[5]:


get_ipython().system('pip install --user gensim ')
get_ipython().system('pip install --user gensim pyLDAvis wordlcoud')


# # Read the Dataset into a dataframe with apt encoding

# In[66]:


import os, pandas as pd
# Add asset from file system
complaints_df = pd.read_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/Complaints_All_2018_Consolidated_Final_For_upload.csv',
                            encoding='ISO-8859-1')
complaints_df.columns



# # Explore Data

# In[67]:


complaints_df.head(3)


# # Combine the text corpus to create the final dataframe

# In[68]:


columns= ['ConcernDescription','IssueDescription','ResolutionDescription']
keywords_df = complaints_df[columns]
keywords_df['keyword_corpus'] = keywords_df[columns[0]]+"."+keywords_df[columns[1]]+"."+keywords_df[columns[2]]
keywords_df['keyword_corpus'].head(3)


# In[69]:


keywords_df['keyword_corpus'][0]


# # Clean the data with regex
# 

# In[70]:


import re
from nltk.tokenize import word_tokenize


# In[71]:


#remove email
def noemail(string):
    string = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+',' ',string)
    return string


# In[72]:


#clean the strings
def cleanstring(string):
    cstring = []
    for x in string:
        if x.isalpha():
            cstring.append(x)
        elif x ==' ':
            cstring.append(x)
        else:
            cstring = cstring
    cstring = ''.join(cstring)
    return cstring


# In[73]:


def preprocess(doc):
    doc = nltk.word_tokenize(doc)
    doc = nltk.pos_tag(doc)
    return doc


# In[74]:


# remove names
def removeNNP(string):
    doc = []
    for wordy in string:
        if wordy[1]=='NNP':
            doc = doc
        else:
            doc.append(wordy[0])
    return(doc)


# In[75]:


#create new column for tokens
keywords_df['keyword_tokens'] = keywords_df['keyword_corpus'].copy()


# In[76]:


#fill blanks: NOTE: this is not needed with Nijesh's update
keywords_df['keyword_tokens'] = keywords_df['keyword_corpus'].where((pd.notnull(keywords_df['keyword_tokens'])), "nblank")


# In[77]:


#remove emails
keywords_df['keyword_tokens'] = keywords_df['keyword_tokens'].apply(noemail)


# In[78]:


#clean to only is alpha
keywords_df['keyword_tokens'] = keywords_df['keyword_tokens'].apply(cleanstring)


# # Remove the stopwords using Spacy
# 

# In[79]:


#! pip install spacy
#! python -m spacy download en


# In[80]:


import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en")
nlpstop = nlp.Defaults.stop_words


# In[81]:


from gensim.parsing.preprocessing import remove_stopwords


# In[82]:


def removestop(docs):
    remaining = remove_stopwords(docs)
    return remaining


# In[83]:


keywords_df['keyword_tokens'] = keywords_df['keyword_tokens'].apply(removestop)


# # Tokenise the data

# In[84]:


def tokens(docs):
    tokes = nltk.word_tokenize(docs)
    return tokes


# In[85]:


#tokenize
keywords_df['keyword_tokens'] = keywords_df['keyword_tokens'].apply(tokens)


# In[86]:


print(keywords_df['keyword_tokens'].head(5))


# In[ ]:





# In[ ]:




