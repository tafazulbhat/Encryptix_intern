#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IRIS FLOWER CLASSIFICATION
import pandas as pd
data=pd.read_csv('iris_dataset.csv')
data.head()


# In[2]:


data.shape


# In[3]:


data['species'].value_counts()


# In[5]:


x=data.iloc[:,[0,1,2,3]]
x


# In[6]:


x.values


# In[7]:


y=data.iloc[:,4]
y


# In[8]:


y.values


# In[10]:


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y=lb.fit_transform(y)
y


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
cf=RandomForestClassifier()
cf.fit(x_train,y_train)


# In[15]:


accuracy=cf.score(x_test,y_test)
accuracy


# In[ ]:




