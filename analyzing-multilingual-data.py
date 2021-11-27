#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


pip install langid


# In[3]:


pip install spacy


# In[4]:


pip install -U pip setuptools wheel


# In[5]:


pip install -U spacy


# In[6]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[7]:


# import libraries we'll use
import spacy # fast NLP
import pandas as pd # dataframes
import langid # language identification (i.e. what language is this?)
from nltk.classify.textcat import TextCat # language identification from NLTK
from matplotlib.pyplot import plot # not as good as ggplot in R :p


# In[8]:


# read in our data
tweetsData = pd.read_csv("all_annotated.tsv", sep = "\t")

# check out some of our tweets
tweetsData['Tweet'][0:15]


# In[9]:


#from spacy.en import Englis
#from spacy import English
import spacy
# create a Spacy document of our tweets
# load an English-language Spacy model
nlp = spacy.load("en_core_web_sm")
# apply the english language model to our tweets
doc = nlp(' '.join(tweetsData['Tweet']))


# In[10]:


sorted(doc, key=len, reverse=True)[0:5]


# In[11]:


sorted(doc, key=len, reverse=True)[6:10]


# In[12]:


# summerize the labelled language
tweetsData['Tweet'][0:5].apply(langid.classify)


# In[13]:


import nltk


# In[14]:


nltk.download('crubadan')


# In[15]:


# N-Gram-Based Text Categorization
tc = TextCat()

# try to identify the languages of the first five tweets again
tweetsData['Tweet'][0:5].apply(tc.guess_language)


# In[16]:


# get the language id for each text
ids_langid = tweetsData['Tweet'].apply(langid.classify)

# get just the language label
langs = ids_langid.apply(lambda tuple: tuple[0])

# how many unique language labels were applied?
print("Number of tagged languages (estimated):")
print(len(langs.unique()))

# percent of the total dataset in English
print("Percent of data in English (estimated):")
print((sum(langs=="en")/len(langs))*100)


# In[17]:


# convert our list of languages to a dataframe
langs_df = pd.DataFrame(langs)

# count the number of times we see each language
langs_count = langs_df.Tweet.value_counts()

# horrible-looking barplot (I would suggest using R for visualization)
langs_count.plot.bar(figsize=(20,10), fontsize=20)


# In[18]:


print("Languages with more than 400 tweets in our dataset:")
print(langs_count[langs_count > 400])

print("")

print("Percent of our dataset in these languages:")
print((sum(langs_count[langs_count > 400])/len(langs)) * 100)


# In[19]:


get_ipython().system('python -m spacy download es')


# In[20]:


# get a list of tweets labelled "es" by langid
spanish_tweets = tweetsData['Tweet'][langs == "es"]

# load a Spanish-language Spacy model
#from spacy.es import Spanish
nlp_es = spacy.load("es_core_news_sm")
#nlp_es = Spanish(path=None)

# apply the Spanish language model to our tweets
doc_es = nlp_es(' '.join(spanish_tweets))

# print the longest tokens
sorted(doc_es, key=len, reverse=True)[0:5]

