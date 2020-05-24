#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits #contains many small images


# In[2]:


digits = load_digits() #loading training set


# In[3]:


dir(digits) #explore


# In[5]:


digits.data[0] # images expand in array called numeric data for image 0


# In[6]:


digits.data[2] # images expand in array called numeric data for image 2


# In[7]:


#showing the image in form of image form
plt.gray()
plt.matshow(digits.images[0])


# In[8]:


#showing the image in form of image form
plt.gray()
plt.matshow(digits.images[4])


# In[18]:


# printing all images
plt.gray()
for i in range(7):
    plt.matshow(digits.images[i])


# In[12]:


digits.target[0:7] #target set as target variables


# In[14]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)


# In[19]:


len(x_train)


# In[20]:


len(x_test)


# In[25]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[26]:


model.fit(x_train, y_train) # training purpose


# In[27]:


model.fit(x_test, y_test) #test purpose


# In[28]:


model.score(x_test, y_test) #for testing purpose


# In[33]:


plt.matshow(digits.images[67])


# In[34]:


digits.target[67]


# In[35]:


model.predict([digits.data[67]]) # predicting the target variable


# In[37]:


model.predict(digits.data[0:5])


# In[45]:


#creating conclusion matrix
y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)
cm


# In[50]:


import seaborn as sns
plt.figure(figsize = (10,7))
sns.heatmap=(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




