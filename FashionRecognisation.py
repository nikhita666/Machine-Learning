#!/usr/bin/env python
# coding: utf-8

# In[53]:


import tensorflow as tf
print(tf.__version__)


# In[54]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True


# In[55]:


callbacks = myCallback()


# In[56]:


mnist = tf.keras.datasets.fashion_mnist


# In[57]:


(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# In[58]:


import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


# In[59]:


training_images  = training_images / 255.0
test_images = test_images / 255.0


# In[60]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# In[61]:


model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])


# In[62]:


model.evaluate(test_images, test_labels)


# In[63]:


classifications = model.predict(test_images)


# In[64]:


print(classifications[0])
print(test_labels[0])

