#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Import Dataset/Exploratory Data Analysis

# In[4]:


#r = 192500
#from math import radians
#dist = r * radians(12)
#print(dist)


# In[5]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[6]:


train


# In[7]:


test


# In[8]:


train.head()


# In[9]:


test.head()


# In[10]:


train.info()


# In[11]:


test.info()


# In[12]:


train.describe()


# In[13]:


test.describe()


# In[15]:


train.isnull().sum()


# In[16]:


test.isnull().sum()


# In[17]:


train.shape


# In[18]:


test.shape


# # Data Visualization

# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# # Barchart with categorical features

# In[22]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    
    dead = train[train['Survived']==0][feature].value_counts()
    
    df = pd.DataFrame([survived, dead])
    
    df.index = ['Survived', 'Dead']
    
    df.plot(kind='bar', stacked=True, figsize=(10, 5))


# In[23]:


bar_chart('Sex')


# In[25]:


bar_chart('Pclass')


# In[26]:


bar_chart('SibSp')


# In[28]:


bar_chart('Parch')


# In[27]:


bar_chart('Embarked')


# In[29]:


train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[30]:


train['Title'].value_counts()


# In[31]:


test['Title'].value_counts()


# # Title Mapping

# In[33]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3,
                 "Col": 3, "Major": 3, "Mlle": 3,
                 "Countess": 3, "Ms": 3, "Lady": 3, 
                 "Jonkheer": 3, "Don": 3, "Dona": 3,
                 "Mme": 3, "Capt": 3, "Sir": 3 }
for dataset in train_test_data :
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[34]:


train.head()


# In[35]:


test.head()


# In[36]:


bar_chart('Title')


# # Delete Unnecessary Feature

# In[37]:


train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[38]:


train.head()


# In[39]:


test.head()


# # Mapping Men/Women

# In[40]:


sex_mapping = {
    "male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[41]:


train.head()


# In[42]:


test.head()


# In[43]:


bar_chart('Sex')


# # Age Missing

# In[44]:


train["Age"].fillna(train.groupby("Title")['Age'].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")['Age'].transform("median"), inplace=True)


# In[45]:


train.groupby("Title")["Age"].transform("median")


# In[46]:


train.head()


# In[47]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

plt.show()


# In[48]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

plt.xlim(0, 20)


# In[49]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

plt.xlim(20, 30)


# In[50]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

plt.xlim(30, 40)


# In[51]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

plt.xlim(40, 60)


# In[52]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

plt.xlim(60, 80)


# In[53]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

plt.xlim(80, 100)


# In[54]:


train.info()


# In[55]:


test.info()


# In[56]:


for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16 , 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62 , 'Age'] = 4
                
    


# In[57]:


train.head()


# In[58]:


bar_chart('Age')


# # Embarked

# In[60]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st Class', '2nd Class', '3rd Class']
df.plot(kind='bar', stacked=True, figsize=(10,5))


# In[61]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[62]:


train.head()


# In[64]:


embarked_mapping = {
    "S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# # Fare

# In[66]:


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[67]:


train.head()


# In[70]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()

plt.show()


# In[71]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()

plt.xlim(0, 20)


# In[72]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()

plt.xlim(0, 30)


# # Fare Mapping

# In[73]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare']=1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare']=2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3


# In[74]:


train.head()


# # Cabin

# In[75]:


train.Cabin.value_counts()


# In[76]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[77]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st Class', '2nd Class', '3rd Class']

df.plot(kind='bar', stacked=True, figsize=(10,5))


# # Cabin Map

# In[78]:


cabin_mapping = {"A":0, "B":0.4, "C":0.8, "D":1.2, "E":1.6, "F":2.0, "G":2.4, "T":2.8}

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[79]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[80]:


train.head()


# In[81]:


test.head()


# # Family Members

# In[82]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + train["Parch"] + 1


# In[85]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0, train['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)


# # Family Mapping

# In[86]:


family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[87]:


train.head()


# # Unnecessary Feature Drop

# In[88]:


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop, axis = 1)

train = train.drop(['PassengerId'], axis=1)


# In[89]:


train_data = train.drop('Survived', axis = 1)
target = train['Survived']

train_data.shape, target.shape


# In[90]:


train_data.head()


# In[91]:


train_data.isnull().sum()


# # ML Model

# # Decision Tree & Random Forest

# In[92]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np


# In[93]:


train.info()


# In[94]:


test.info()


# # Cross Validation

# In[95]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# # Decision Tree

# In[96]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[97]:


round(np.mean(score)*100, 2)


# # Random Forest

# In[98]:


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score=cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[99]:


round(np.mean(score)*100, 2)


# # Kaggle Upload

# In[102]:


clf = RandomForestClassifier(n_estimators=13)
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[103]:


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": prediction
})

submission.to_csv('submission.csv', index=False)


# # Show Submission File

# In[ ]:


submission = pd.read_csv('submission.csv')

submission.head()

