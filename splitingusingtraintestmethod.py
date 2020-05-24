#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
df=pd.read_csv('G:\\car.csv')
df


# In[31]:


df.dtypes


# In[32]:


st = pd.DataFrame(df)
st


# In[35]:


df.head()


# In[38]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


plt.scatter(df['milege'],df['price'])


# In[43]:


plt.scatter(df['age'],df['price'])


# In[44]:


x=df[['milege','age']]


# In[45]:


y=df['price']


# In[46]:


x


# In[49]:


y


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=10)


# In[52]:


len(x_train)


# In[53]:


len(y_train)


# In[57]:


x_train


# In[58]:


y_train


# In[59]:


x_train


# In[60]:


y_train


# In[61]:


from sklearn.linear_model import LinearRegression
clf=LinearRegression()


# In[62]:


clf.fit(x_train,y_train)


# In[64]:


clf.predict(x_test)


# In[65]:


y_test


# In[66]:


clf.score(x_test,y_test)


# In[ ]:




