#!/usr/bin/env python
# coding: utf-8

# In[38]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[39]:


df = pd.read_csv('chirps.csv')


# In[40]:


df


# In[41]:


df.shape


# In[42]:


df.isnull().sum()


# # Separate X, Y

# In[43]:


#x = df.drop('Chirps_15s', axis = 1)
x = df['Temp_C']


# In[44]:


x


# In[45]:


y = df['Chirps_15s']


# In[46]:


y


# In[47]:


plt.xlabel('Cricket Chirp / 15 Seconds')


# In[48]:


plt.ylabel('Temperature / Celsius')


# In[49]:


plt.xlabel('Cricket Chirp / 15 Seconds')
plt.ylabel('Temperature / Celsius')
plt.title('Cricket Chirp and Temperature')
plt.grid(True)
plt.plot(x, y, 'ro')
plt.show()


# In[53]:


plt.plot(df.Temp_C)


# # Seaborn Library

# In[51]:


import seaborn as sns; sns.set(color_codes=True)
g = sns.lmplot(x="Chirps_15s", y = "Temp_C", data = df)
plt.title('Cricket Chirp and Temperature')
plt.show()


# In[ ]:




