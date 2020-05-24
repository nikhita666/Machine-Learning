#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as plt
from sklearn import linear_model


# In[43]:


df=pd.read_csv("G:\\SQL\\flat.csv")
df


# In[44]:


#df['bedrooms'].median()
import math
median_bd=math.floor(df.bedrooms.median())
median_bd


# In[45]:


df.bedrooms.fillna(median_bd)


# In[25]:


df.shape


# In[49]:


y=df.iloc[:,:1].values
x=df.iloc[:,1:].values


# In[50]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[51]:


from sklearn.linear_model import LinearRegression


# In[52]:


reg=LinearRegression()


# In[53]:


reg.fit(x_train,y_train)


# In[54]:


pred=reg.predict(x_test)
pred


# In[55]:


y_test


# In[ ]:




