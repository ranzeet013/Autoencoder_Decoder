#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# Importing libraries is the process of loading external code libraries into your Python program to access their pre-defined functions, classes, and other resources. Libraries are collections of code that provide additional functionality beyond what is available in the core Python language.

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# # Loading MNIST Data From keras.datasets 

# In[4]:


from tensorflow.keras.datasets import mnist


# # Splitting Dataset

# Train-Test Split: In some cases, a separate validation set may not be necessary. Instead, the dataset is split into a training set and a test set only. The training set is used for model training, and the test set is used for final model evaluation. The split ratio is usually around 80-90% for training and 10-20% for testing.

# In[5]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[7]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[8]:


plt.imshow(x_train[5])


# In[9]:


x_train.max()


# In[10]:


x_train = x_train / 225
x_test = x_test / 225


# # Noising Image 

# Gaussian noise, also known as white noise or Gaussian white noise, is a type of random noise that follows a Gaussian distribution. It is characterized by its mean and standard deviation, which determine the center and spread of the distribution, respectively. Gaussian noise is commonly used in various fields, including signal processing and machine learning, for different purposes such as data augmentation, regularization, or modeling uncertainty.

# In[12]:


from tensorflow.keras.layers import GaussianNoise


# In[13]:


sample = GaussianNoise(0.2)


# In[14]:


noisey = sample(x_test[0:2], training = True)


# In[15]:


noisey


# In[17]:


plt.imshow(x_test[1])


# In[18]:


plt.imshow(noisey[1])


# In[20]:


tf.random.set_seed = 101
np.random_seed = 101


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape


# # Building Encoder Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.
# To build an encoder model, you can use a convolutional neural network (CNN) architecture. The encoder's role is to extract relevant features from the input data.

# In[23]:


encoder = Sequential()
encoder.add(GaussianNoise(0.2))
encoder.add(Flatten(input_shape = (28, 28)))
encoder.add(Dense(400, activation = 'relu'))
encoder.add(Dense(200, activation = 'relu'))
encoder.add(Dense(100, activation = 'relu'))
encoder.add(Dense(50, activation = 'relu'))
encoder.add(Dense(25, activation = 'relu'))


# In[28]:


decoder = Sequential()
decoder.add(Dense(50, input_shape = [25], activation = 'relu'))
decoder.add(Dense(100, activation = 'relu'))
decoder.add(Dense(200, activation = 'relu'))
decoder.add(Dense(400, activation = 'relu'))
decoder.add(Dense(28*28, activation = 'sigmoid'))
decoder.add(Reshape([28, 28]))


# In[29]:


remove_noise = Sequential([encoder, decoder])


# # Compiling Model

# Compiling a model in deep learning involves configuring the training process by specifying the optimizer, loss function, and metrics to be used. The compilation step is necessary before training the model.

# In[30]:


remove_noise.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training Model

# Training a deep learning model involves iteratively updating the model's parameters or weights using a training dataset. The process aims to minimize the loss function by adjusting the model's parameters through an optimization algorithm.

# In[31]:


remove_noise.fit(x_train,
                 x_train, 
                 validation_data = (x_test, x_test),
                 epochs = 10)


# In[32]:


loss = pd.DataFrame(remove_noise.history.history)


# In[33]:


loss.plot()


# In[35]:


loss[['loss', 'val_loss']].plot()


# In[36]:


loss[['accuracy', 'val_accuracy']].plot()


# # Denoising Image

# Denoising an image refers to the process of reducing or removing unwanted noise or disturbances from an image while preserving the essential details and structures. Image denoising is a common task in image processing and computer vision, and it aims to improve the visual quality and enhance the interpretability of images.

# In[39]:


noise_image_top_20 = sample(x_test[0:20], training = True)


# In[41]:


denoised = remove_noise(noise_image_top_20[0:20])


# In[48]:


n = 1
print("Orgiginal Image: ")
plt.imshow(x_test[n])
plt.show()
print("Noisey Image: ")
plt.imshow(noise_image_top_20[n])
plt.show()
print("Denoised Image: ")
plt.imshow(denoised[n])
plt.show()


# In[ ]:




