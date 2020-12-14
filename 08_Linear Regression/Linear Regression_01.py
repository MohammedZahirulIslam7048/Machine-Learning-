#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Dataset

# In[2]:


df = pd.read_csv('dhaka homeprices.csv')


# In[3]:


df


# In[4]:


df.shape


# In[7]:


#df.head()
df.head(3)


# # Check Null Value

# In[9]:


#df.isnull().any()
df.isnull().sum()


# # Separate X, Y

# In[10]:


x = df[['area']]


# In[11]:


x


# In[12]:


y = df['price']


# In[13]:


y


# # Visualization (Scatter Plot)

# In[33]:


'''
plt.scatter(df['area'], df['price'], marker='+', color = 'blue')
#plt.scatter(df['area'], df['price'])
plt.plot(df.area, reg.predict(df[['area']]))
plt.xlabel('Area in Sq. ft')
plt.ylabel('Price in taka')
plt.title('Home prices in Dhaka')
'''


# # Split Train Test

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .20, random_state = 1)


# In[25]:


xtrain


# In[26]:


xtest


# In[27]:


ytrain


# In[28]:


ytest


# # Algorithm

# In[29]:


from sklearn.linear_model import LinearRegression


# # Create Object

# In[30]:


reg = LinearRegression()


# In[34]:


reg.fit(xtrain, ytrain)


# # Predict Price

# In[35]:


reg.predict(xtest)


# # Score

# In[40]:


reg.score(xtest, ytest)


# # Plot

# In[36]:


plt.scatter(df['area'], df['price'], marker='+', color = 'blue')
#plt.scatter(df['area'], df['price'])
plt.plot(df.area, reg.predict(df[['area']]))
plt.xlabel('Area in Sq. ft')
plt.ylabel('Price in taka')
plt.title('Home prices in Dhaka')


# # Predict With Indivisual Value

# In[37]:


reg.predict([[3500]])


# In[38]:


reg.coef_


# In[39]:


reg.intercept_


# # Price for New Area

# In[43]:


n = input('Enter Area : ')
array = np.array(n)
array2 = array.astype(np.float)
value = ([[array2]])
result = reg.predict(value)
home_price = np.array(result)
home_price = home_price.item()
print('Predicted home price :', home_price)


# In[ ]:




