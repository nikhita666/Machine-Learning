#!/usr/bin/env python
# coding: utf-8

# In[12]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[13]:


df = pd.read_csv("G:\EXCEL SHEETS\GOOG (1).csv")
df


# In[14]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in $', fontsize=18)
plt.show


# In[15]:


data_train = df[df['Date']<'2019-01-01'].copy()
train_da = data_train.drop(['Date'], axis = 1)
train_da


# In[16]:


scaler = MinMaxScaler()
train_da = scaler.fit_transform(train_da)
train_da


# In[17]:


x_train = []
y_train = []

for i in range(60, train_da.shape[0]):
    x_train.append(train_da[i-60:i])
    y_train.append(train_da[i, 0])
    if i<=60:
        print(x_train)
        print(y_train)


# In[18]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[19]:


x_train.shape, y_train.shape


# In[38]:


model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences= True, input_shape= (x_train.shape[1],6)))
model.add(Dropout(0.2))

model.add(LSTM(units=70, activation='relu', return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units=90, activation='relu', return_sequences= True))
model.add(Dropout(0.6))

model.add(LSTM(units=130, activation='relu', return_sequences= True))
model.add(Dropout(0.8))

model.add(LSTM(units=150, activation='relu'))
model.add(Dropout(0.9))

model.add(Dense(units=1))


# In[39]:


model.summary()


# In[40]:


model.compile(optimizer='adam', loss = 'mean_squared_error')


# In[43]:


history = model.fit(x_train, y_train, batch_size=1, epochs=20)


# In[44]:


plt.plot(history.history['loss'])
plt.show()


# In[26]:


data_test = df[df['Date']>='2019-01-01'].copy()
test_da = data_test.drop(['Date', 'Adj Close'], axis = 1)
data_test


# In[27]:


past_60_days = data_train.tail(60)


# In[28]:


df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date'], axis = 1)
df


# In[29]:


inputs = scaler.transform(df)
inputs


# In[30]:


x_test = []
y_test = []
for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])


# In[31]:


x_test, y_test = np.array(x_test), np.array(y_test)


# In[32]:


x_test.shape, y_test.shape


# In[33]:


y_prediction = model.predict(x_test)
scale = 1/8.18605127e-04
y_prediction = y_prediction*scale
y_test = y_test*scale


# In[34]:


plt.figure(figsize=(16,8))
plt.plot(y_test, color='red', label='Real Google stock price')
plt.plot(y_prediction, color='blue', label='predicted GOOG stock price')
plt.title('Google Stock Price Rrediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend
plt.show()


# In[ ]:




