#!/usr/bin/env python
# coding: utf-8

# # Import Ilbrary

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Dataset

# In[10]:


df = pd.read_csv('online.csv')


# In[11]:


df


# In[12]:


df.shape


# # Check Null Value

# In[13]:


df.isnull().sum()


# # Calculate Null Value

# In[18]:


# Calculate Mean
missing = df.Administration.mean()
missing


# In[19]:


# Add missing value to Dataset
df.Administration = df.Administration.fillna(missing)


# In[20]:


df


# # Separate X,Y

# In[21]:


x = df.drop(['Profit'], axis = 1)


# In[22]:


x


# In[23]:


y = df['Profit']


# In[24]:


y


# # Convert String to Numeric

# In[26]:


city = pd.get_dummies(x['Area'], drop_first = True)


# In[27]:


city


# # Drop Area Column

# In[28]:


x = x.drop('Area', axis=1)


# In[29]:


x


# # Concatenation Area Column

# In[30]:


x = pd.concat([x,city], axis=1)


# In[31]:


x


# # Spilting Dataset into the Training and Test

# In[32]:


#Import Library
from sklearn.model_selection import train_test_split


# In[33]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 0)


# # Fitting Multiple Linear Regression to the Training Set

# In[36]:


# Import Library
from sklearn.linear_model import LinearRegression


# # Create Object

# In[37]:


regressor = LinearRegression()


# In[42]:


regressor.fit(xtrain, ytrain)


# In[52]:


xtest


# In[53]:


ytrain


# # Predicting the Test Set Results

# In[54]:


pred = regressor.predict(xtest)


# In[55]:


pred


# In[56]:


regressor.score(xtest, ytest)


# # R Squared Value

# In[57]:


# Import Library
from sklearn.metrics import r2_score
score = r2_score(ytest, pred)
score


# In[ ]:





# In[ ]:




