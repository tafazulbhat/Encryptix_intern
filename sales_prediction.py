#!/usr/bin/env python
# coding: utf-8

# In[30]:


#SALES PREDICTION
import pandas as pd
data=pd.read_csv('sales_dataset.csv')
data.head()


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data)
plt.show()


# In[32]:


plt.figure(figsize=(8, 5))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()


# In[33]:


x = data.iloc[:,[0,1,2]] 
y = data.iloc[:,3] 


# In[34]:


x


# In[35]:


y


# In[36]:


x.values


# In[37]:


y.values


# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[39]:


from sklearn.linear_model import LinearRegression
lg=LinearRegression()
lg.fit(x_train, y_train)


# In[40]:


predictions= lg.predict(x_test)


# In[41]:


from sklearn.metrics import mean_squared_error, r2_score
performance = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", performance)
print("R² Score:", r2)


# In[44]:


print("Intercept:", lg.intercept_)
print("Coefficients:", lg.coef_)


# In[ ]:




