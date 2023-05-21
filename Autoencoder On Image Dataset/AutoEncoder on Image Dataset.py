#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# Importing libraries is an essential step in any data analysis or machine learning project. These libraries provide various functions and tools to manipulate, visualize, and analyze data efficiently. Here are explanations of some popular data analysis libraries:

# Pandas: Pandas is a powerful and widely used library for data manipulation and analysis. It provides data structures like DataFrames and Series, which allow you to store and manipulate tabular data. Pandas offers a wide range of functions for data cleaning, filtering, aggregation, merging, and more.

# NumPy: NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides efficient data structures like arrays and matrices and a vast collection of mathematical functions. NumPy enables you to perform various numerical operations on large datasets, such as element-wise calculations, linear algebra, Fourier transforms, and random number generation.

# Matplotlib: Matplotlib is a popular plotting library that enables you to create a wide range of static, animated, and interactive visualizations. It provides a MATLAB-like interface and supports various types of plots, including line plots, scatter plots, bar plots, histograms, and more. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing MNIST Dataset From keras.datasets

# In[3]:


from tensorflow.keras.datasets import mnist


# # Splitting Dataset

# Splitting a dataset is the process of dividing it into separate subsets for training, validation, and testing. Here's a the common ways to split a dataset:

# Train-Test Split: In some cases, a separate validation set may not be necessary. Instead, the dataset is split into a training set and a test set only. The training set is used for model training, and the test set is used for final model evaluation. The split ratio is usually around 80-90% for training and 10-20% for testing.

# In[4]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[5]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[6]:


plt.imshow(x_train[3])


# In[7]:


x_train.min()


# In[8]:


x_train.max()


# In[9]:


x_train = x_train / 255
x_test = x_test / 225


# # Importing Deep Learning Libraries

# Importing deep learning libraries allows you to access and utilize their functionalities and APIs for building, training, and evaluating deep learning models. Here's a of some popular deep learning libraries:

# TensorFlow: TensorFlow is an open-source deep learning library developed by Google. It provides a flexible and comprehensive ecosystem for building and training various types of neural networks. TensorFlow uses a symbolic dataflow graph to define and execute computational operations, making it efficient for both research and production use cases

# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import SGD


# # Building Encoder Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# To build an encoder model, you can use a convolutional neural network (CNN) architecture. The encoder's role is to extract relevant features from the input data.

# In[14]:


encoder = Sequential()
encoder.add(Flatten(input_shape = [28, 28]))
encoder.add(Dense(400, activation = 'relu'))
encoder.add(Dense(200, activation = 'relu'))
encoder.add(Dense(100, activation = 'relu'))
encoder.add(Dense(50, activation = 'relu'))
encoder.add(Dense(25, activation = 'relu'))


# # Building Decoder Model

# In[16]:


decoder = Sequential()
decoder.add(Dense(50, input_shape = [25], activation = 'relu'))
decoder.add(Dense(100, activation = 'relu'))
decoder.add(Dense(200, activation = 'relu'))
decoder.add(Dense(100, activation = 'relu'))
decoder.add(Dense(28*28, activation = 'sigmoid'))
decoder.add(Reshape([28, 28]))


# In[17]:


autoencoder = Sequential([encoder, decoder])


# # Compiling Model

# Compiling a model in deep learning involves configuring the training process by specifying the optimizer, loss function, and metrics to be used. The compilation step is necessary before training the model.

# In[18]:


autoencoder.compile(optimizer = SGD(lr = 1.5), loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training The Model

# Training a deep learning model involves iteratively updating the model's parameters or weights using a training dataset. The process aims to minimize the loss function by adjusting the model's parameters through an optimization algorithm.

# In[20]:


autoencoder.fit(x_train,
                x_train, 
                validation_data = (x_test, x_test),
                epochs = 10)


# In[21]:


loss = pd.DataFrame(autoencoder.history.history)


# In[22]:


loss.plot()


# # Predicting on dataset

# In[23]:


encoded_image = autoencoder.predict(x_test[:5])


# In[25]:


plt.imshow(encoded_image[4])


# In[26]:


plt.imshow(x_test[4])


# # Denoising Images

# Gaussian noise, also known as white noise or Gaussian white noise, is a type of random noise that follows a Gaussian distribution. It is characterized by its mean and standard deviation, which determine the center and spread of the distribution, respectively. Gaussian noise is commonly used in various fields, including signal processing and machine learning, for different purposes such as data augmentation, regularization, or modeling uncertainty.

# In[28]:


from tensorflow.keras.layers import GaussianNoise


# In[29]:


sample = GaussianNoise(0.2)


# In[31]:


noisey = sample(x_test[0:2], training = True)


# In[32]:


noisey


# In[38]:


plt.imshow(x_test[1])


# In[39]:


plt.imshow(noisey[1])


# In[ ]:




