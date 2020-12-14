#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model


# # Dataset

# In[2]:


df = pd.read_csv('car_risk_multiple_value.csv')


# In[3]:


df


# # See Every Column

# In[5]:


#df.speed
#df.car_age
#df.experience
#df.risk


# # Check Null Value

# In[6]:


df.isnull().sum()


# # Calculate Mean/ Median for Null Value

# In[16]:


#Mean
exp_fit_mean = df.experience.mean()


# In[18]:


exp_fit_mean


# In[19]:


#Median
exp_fit = df.experience.median()


# In[20]:


exp_fit


# # Fit Null Value Into Dataset

# In[21]:


df.experience = df.experience.fillna(exp_fit)


# In[22]:


df


# In[23]:


df.experience


# # Create Object

# In[24]:


reg = linear_model.LinearRegression()


# # Train Data

# In[25]:


reg.fit(df[['speed', 'car_age', 'experience']], df.risk)


# # Predict Result

# In[26]:


reg.predict([[160, 10, 5]])


# # Check by Equation

# In[27]:


#Coefficient
reg.coef_


# In[28]:


#Intercept
reg.intercept_


# In[29]:


0.33059217 * 160 + 1.61053246 * 10 + -6.20772074 * 5 + 33.410000910435905


# In[ ]:




