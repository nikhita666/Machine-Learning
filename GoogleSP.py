#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[95]:


import pandas as pd
data = pd.read_csv("G:\EXCEL SHEETS\GOOG (1).csv")
data


# In[96]:


data_train = data[data['Date']<'2019-01-01'].copy()
train_d = data_train.drop(['Date', 'Adj Close'], axis = 1)
train_d


# In[97]:


scaler = MinMaxScaler()
train_d = scaler.fit_transform(train_d)
train_d


# In[98]:


x_train = []
y_train = []
for i in range(60, train_d.shape[0]):
    x_train.append(train_d[i-60:i])
    y_train.append(train_d[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape, y_train.shape


# In[68]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[99]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Dropout
    


# In[100]:


regressior = Sequential()

regressior.add(LSTM(units=50, activation='relu', return_sequences= True, input_shape= (x_train.shape[1],5)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units=60, activation='relu', return_sequences= True))
regressior.add(Dropout(0.3))

regressior.add(LSTM(units=80, activation='relu', return_sequences= True))
regressior.add(Dropout(0.4))

regressior.add(LSTM(units=120, activation='relu'))
regressior.add(Dropout(0.5))

regressior.add(Dense(units=1))


# In[81]:


model = Sequential()
model.add(LSTM(60, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(60, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[101]:


regressior.summary()


# In[102]:


regressior.compile(optimizer='adam', loss = 'mean_squared_error')


# In[103]:


regressior.fit(x_train, y_train, epochs=15, batch_size=32)


# In[108]:


past_60_days = data_train.tail(60)


# In[111]:


df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)
df


# In[112]:


inputs = scaler.transform(df)
inputs


# In[113]:


x_test = []
y_test = []
for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test.shape, y_test.shape


# In[114]:


y_pred = regressior.predict(x_test)


# In[115]:


scale = 1/8.18605127e-04
y_pred = y_pred*scale
y_test = y_test*scale


# In[116]:


plt.figure(figsize=(14,5))
plt.plot(y_test, color='red', label='Real Google stock price')
plt.plot(y_pred, color='blue', label='predicted GOOG stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend
plt.show()


# In[ ]:




