#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np


# In[13]:


population = np.random.randn(30)
print(population)


# In[14]:


sample = np.random.choice(population, 20)
print(sample)


# In[15]:


result_population = np.var(population)
result_sample = np.var(sample)


# In[16]:


print('Population : ', result_population)
print('Sample : ', result_sample)


# In[ ]:




