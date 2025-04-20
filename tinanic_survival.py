#!/usr/bin/env python
# coding: utf-8

# In[9]:


#TITANIC SURVIVAL PREDICTION
import pandas as pd
data=pd.read_csv('Titanic-Dataset.csv')


# In[10]:


data.head()


# In[11]:


data.shape


# In[12]:


data.info()


# In[13]:


print(data['Embarked'].mode())


# In[14]:


data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)


# In[15]:


data['Age'].fillna(data['Age'].median(),inplace=True)


# In[16]:


data.info()


# In[17]:


d1=data.drop(columns=['PassengerId','Name','Ticket','Cabin'])


# In[18]:


d1.head()


# In[19]:


d1['Survived'].value_counts()


# In[20]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
d1['Sex'] = label_encoder.fit_transform(d1['Sex'])
d1['Embarked'] = label_encoder.fit_transform(d1['Embarked'])


# In[21]:


d1.head()


# In[22]:


x=d1.drop(columns='Survived')
x.head()


# In[23]:


y=d1['Survived']
y.head()


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[27]:


x.shape,x_train.shape,x_test.shape


# In[28]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)


# In[29]:


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print(y_predict)


# In[37]:


testing_accuracy = accuracy_score(y_test, y_predict)
print('testing data accuracy score =',testing_accuracy)


# In[ ]:




