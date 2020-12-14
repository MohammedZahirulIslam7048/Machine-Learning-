#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import numpy as np
import pandas as pd


# # Dataset

# In[2]:


df = pd.read_csv('carprices.csv')


# In[3]:


df


# In[4]:


df.shape


# In[6]:


df.head(3)


# # Preprocessing

# In[7]:


dummies = pd.get_dummies(df['Car Model'])


# In[8]:


dummies


# In[11]:


dummies1 = pd.get_dummies(df['Car Model_1'])


# In[12]:


dummies1


# # Concat

# In[13]:


merged = pd.concat([df, dummies], axis = 'columns')


# In[14]:


merged


# In[18]:


merged.shape


# In[20]:


merged1 = pd.concat([df, dummies1], axis = 'columns')


# In[21]:


merged1


# In[22]:


merged1.shape


# # Delete Column

# In[27]:


final = merged.drop(["Car Model"], axis = 1)


# In[28]:


final


# In[29]:


final1 = final.drop(["Car Model_1"], axis = 1)


# In[30]:


final1


# # Separate X, Y

# In[31]:


x = final1.drop(['Sell Price'], axis = 1)


# In[32]:


x


# In[33]:


y = df['Sell Price']


# # Algorithm (Linear Regression)

# In[34]:


from sklearn.linear_model import LinearRegression


# # Object

# In[35]:


model = LinearRegression()


# # Fit Model

# In[37]:


model.fit(x, y)


# # Score

# In[38]:


model.score(x,y)


# # New Price Predict

# In[39]:


model.predict([[45000, 8, 1, 0, 0, 0]])


# In[ ]:




