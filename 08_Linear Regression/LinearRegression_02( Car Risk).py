#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Dataset

# In[3]:


df = pd.read_csv('car driving risk analysis.csv')


# In[4]:


df


# In[6]:


df.head()


# In[7]:


df.shape


# # Check Null Value

# In[8]:


df.isnull().sum()


# # Separate X, Y

# In[9]:


x = df[['speed']]


# In[10]:


x


# In[13]:


y = df['risk']


# In[14]:


y


# # Split Train, Test

# In[15]:


from sklearn.model_selection import train_test_split


# In[23]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .40, random_state = 1)


# In[27]:


#xtrain
xtest


# In[28]:


#ytrain
ytest


# # Algorithm

# In[29]:


from sklearn.linear_model import LinearRegression


# # Object

# In[30]:


lr = LinearRegression()


# In[31]:


lr.fit(xtrain, ytrain)


# # Score

# In[32]:


lr.score(xtest, ytest)


# # Risk for New Speed

# In[33]:


n = input('New Speed : ')
array = np.array(n)
array2 = array.astype(np.float)
value = [[array2]]
result = lr.predict(value)
risk_new = np.array(result)
risk_new = risk_new.item()
print('Predicted risk :', risk_new)


# # Plot

# In[35]:


plt.scatter(df['speed'], df['risk'], marker='+', color = 'blue')
plt.plot(df.speed, lr.predict(df[['speed']]))
plt.xlabel('Speed in KM/H')
plt.ylabel('Risk in %')
plt.title('Risk analysis with speed')


# In[ ]:




