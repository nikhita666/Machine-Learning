#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv('G:\\insurance1.csv')
df


# In[6]:


from matplotlib import pyplot as plt
plt.scatter(df.age,df.have_insurance,marker='+',color='red')


# In[7]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.have_insurance,test_size=0.4)


# In[15]:


x_test


# In[16]:


x_train


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


model = LinearRegression()


# In[19]:


model.fit(x_train,y_train)


# In[20]:


model.predict(x_test)#model is ready


# In[21]:


model.score(x_test,y_test)#accuracy

