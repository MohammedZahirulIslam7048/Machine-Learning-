#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np


# In[13]:


population_std = np.random.randn(30)


# In[14]:


print('Population : ',population_std)


# In[15]:


sample_std = np.random.choice(population_std, 20)


# In[16]:


print('Sample : ', sample_std)


# In[17]:


result_population = np.std(population_std)


# In[18]:


result_sample = np.std(sample_std)


# In[19]:


print('Population S.D : ', result_population)
print('Sample S.D : ', result_sample)


# In[ ]:




