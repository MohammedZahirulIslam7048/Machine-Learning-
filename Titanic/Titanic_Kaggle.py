#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# # Train Data

# In[28]:


train_data = pd.read_csv('train_.csv')


# In[29]:


train_data


# In[30]:


train_data.head()


# In[31]:


train_data.describe()


# In[32]:


train_data.columns


# In[33]:


features = ["Pclass", "Sex", "SibSp", "Parch"]


# In[34]:


features


# In[35]:


X = train_data[features]


# In[36]:


X


# In[37]:


y = train_data['Survived']


# In[38]:


y


# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)


# In[41]:


model


# In[42]:


model.fit(X, y)


# # Test Data

# In[43]:


test_data = pd.read_csv('test_.csv')


# In[44]:


test_data


# In[45]:


X_test = pd.get_dummies(test_data[features])


# In[46]:


X_test


# In[47]:


predictions = model.predict(X_test)


# In[48]:



print(predictions)


# In[49]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


# In[ ]:





# # MAE

# In[55]:


from sklearn.metrics import mean_absolute_error


# In[56]:


predicted_data = model.predict(X)
print(mean_absolute_error(y, predicted_data))


# In[57]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[61]:


forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(train_X, train_y)
forest_val_predictions = forest_model.predict(val_X)
print('MAE of Random Forests: %f' % (mean_absolute_error(val_y, forest_val_predictions)))


# In[62]:


tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(train_X, train_y)
 
tree_val_predictions = tree_model.predict(val_X)
print('MAE of Decision Tree: %f' % (mean_absolute_error(val_y, tree_val_predictions)))


# In[63]:


# testing
X_test = pd.get_dummies(test_data[features])
predictions = forest_model.predict(X_test)


# In[ ]:


# generate CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': forest_val_predictions})
output.to_csv('my_submission.csv', index=False)


# In[68]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission1.csv', index=False)


# In[ ]:




