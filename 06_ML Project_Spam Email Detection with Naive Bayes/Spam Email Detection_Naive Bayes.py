#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Import Dataset

# In[2]:


df = pd.read_csv('emails.csv')


# In[3]:


df


# In[4]:


df['spam'].value_counts()


# # Drop Same Value

# In[5]:


df.drop_duplicates(inplace = True)


# In[6]:


df


# # Check Null Value

# In[7]:


df.isnull().sum()


# # Separate X, Y

# In[9]:


x = df.text.values


# In[10]:


x


# In[11]:


y = df.spam.values


# In[12]:


y


# # Split Dataset

# In[13]:


# Import Library
from sklearn.model_selection import train_test_split


# In[26]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)


# # Data Preprocessing

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer


# In[28]:


cv = CountVectorizer()


# In[29]:


x_train = cv.fit_transform(xtrain)


# In[30]:


x_train.toarray()


# # Apply Algorithm

# In[31]:


from sklearn.naive_bayes import MultinomialNB


# # Create object

# In[32]:


model = MultinomialNB()


# In[33]:


model.fit(x_train, ytrain)


# In[35]:


x_test = cv.transform(xtest)


# In[36]:


x_test.toarray()


# In[37]:


model.score(x_test, ytest)


# In[38]:


emails = ['1', 'ashad']


# In[39]:


cv_emails = cv.transform(emails)


# In[40]:


model.predict(cv_emails)


# In[ ]:




