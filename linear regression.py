#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 


# In[2]:


df=pd.read_csv('G:\\plot.csv')
df


# In[9]:


df.shape


# In[13]:


df.plot(kind='scatter',x='area',y='price'+)
plt.show()---------------------------------------------+-


# In[14]:


df.plot(kind='box')
plt.show()


# In[17]:


df.corr()


# In[19]:


ar=pd.DataFrame(df['area'])
pr=pd.DataFrame(df['price'])
ar


# In[20]:


ar=pd.DataFrame(df['area'])
pr=pd.DataFrame(df['price'])
pr


# In[21]:


lm = linear_model.LinearRegression()
model=lm.fit(ar,pr)


# In[22]:


model.coef_


# In[24]:


model.intercept_


# In[26]:


model.score(ar,pr)


# In[38]:


ar_new=([[3700]])
pr_new=model.predict(ar_new)
pr_new


# In[44]:


#many more 
X=([2300,3800,4300])
X=pd.DataFrame(X)
Y=model.predict(X)
Y=pd.DataFrame(Y)
df=pd.concat([X,Y] , axis=1 , keys=['area1','price1'])
df


# In[54]:


#visualize the result
df.plot(kind='scatter',x="area",y="price")
#plotting regression plane
#plt.plot(area,model.predict(area),color='red',linewidth=2)
plt.show()


# In[ ]:




