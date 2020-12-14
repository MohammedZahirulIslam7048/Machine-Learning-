#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Dataset 1

# In[2]:


df1 = pd.read_csv('dataset1.csv')


# In[3]:


df1


# In[6]:


df1.shape


# # Dataset 2

# In[4]:


df2 = pd.read_csv('dataset2.csv')


# In[5]:


df2


# In[7]:


df2.shape


# # New Dataset

# In[8]:


df3 = pd.merge(df1, df2, on = 'CustomerID')


# In[9]:


df3


# In[12]:


df3 = pd.merge(df1, df2, on = 'CustomerID', how = 'inner')


# In[13]:


df3


# In[14]:


df3 = pd.merge(df1, df2, on = 'CustomerID', how = 'inner', indicator = True)


# In[15]:


df3


# In[ ]:




