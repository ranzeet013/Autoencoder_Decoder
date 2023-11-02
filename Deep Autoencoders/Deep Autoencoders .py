#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.random import choice
from numpy.linalg import norm


# In[2]:


import pandas as pd 
from pathlib import Path


# In[3]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist


# In[4]:


from sklearn.preprocessing import minmax_scale
from sklearn.manifold import TSNE


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


from scipy.spatial.distance import pdist, cdist
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[7]:


sns.set_style('whitegrid')


# In[8]:


n_classes = 10 


# In[9]:


results_path = Path('results', 'fashion_mnist')
if not results_path.exists():
    results_path.mkdir(parents=True)


# In[10]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# Keras makes it easy to access the 60,000 train and 10,000 test grayscale samples with a resolution of 28 x 28 pixels:

# In[11]:


x_train.shape, x_test.shape


# In[12]:


image_size = 28    
input_size = image_size ** 2


# In[13]:


class_dict = {0: 'T-shirt/top',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sneaker',
              8: 'Bag',
              9: 'Ankle boot'}
classes = list(class_dict.keys())


# ### Sample Images :

# In[15]:


fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (14, 5))
axes = axes.flatten()
for row, label in enumerate(classes):
    label_idx = np.argwhere(y_train  ==  label).squeeze()
    axes[row].imshow(x_train[choice(label_idx)], cmap = 'gray')
    axes[row].axis('off')
    axes[row].set_title(class_dict[row])

fig.suptitle('Fashion MNIST Samples', fontsize = 14)
fig.tight_layout()
fig.subplots_adjust(top=.85)


# ### Reshaping And Normalizing :

# We're taking each image and changing its shape, turning it into a flat, one-dimensional vector with 784 elements. This is a way to represent the image data in a more straightforward format. We're also making sure that the values in this vector are adjusted to a range between 0 and 1, which is a typical approach for handling image data.

# In[16]:


encoding_size = 32 


# In[17]:


def data_prep(x, size=input_size):
    return x.reshape(-1, size).astype('float32')/255


# In[19]:


X_train_scaled = data_prep(x_train)
X_test_scaled = data_prep(x_test)


# In[21]:


X_train_scaled.shape, X_test_scaled.shape


# ### Input Layer :

# In[22]:


input_ = Input(shape = (input_size,), name='Input')


# ### Dense Encoding Layer :

# In[23]:


encoding = Dense(units = encoding_size,
                 activation = 'relu',
                 name = 'Encoder')(input_)


# ### Dense Reconstruction Layer :

# In[26]:


decoding = Dense(units =input_size,
                 activation = 'sigmoid',
                 name = 'Decoder')(encoding)


# ### Autoencoder Model :

# In[27]:


autoencoder = Model(inputs = input_,
                    outputs = decoding,
                    name  ='Autoencoder') 


# In[52]:


autoencoder.summary()


# ### Encoding Model :

# In[29]:


encoder = Model(inputs = input_ ,
                outputs = encoding,
                name = 'Encoder')


# In[30]:


encoder.summary()


# In[31]:


encoded_input = Input(shape = (encoding_size,),
                      name = 'Decoder_Input')


# In[32]:


decoder_layer = autoencoder.layers[-1](encoded_input)


# In[34]:


decoder = Model(inputs = encoded_input,
                outputs = decoder_layer)


# In[51]:


decoder.summary()


# ### Train Autoencoder :

# I'm using an autoencoder model to replicate data. To make it better at this, I'm employing the Adam optimizer, a helpful tool for training. During this training, I'm giving the model the same data as input and what it should produce, so it becomes really good at recreating the original input.

# In[37]:


autoencoder.compile(optimizer = 'adam', loss='mse')


# ### Early Stopping :

# In[38]:


early_stopping = EarlyStopping(monitor = 'val_loss', 
                               min_delta = 1e-5, 
                               patience = 5, 
                               verbose = 0,
                               restore_best_weights = True,
                               mode = 'auto')


# ### Tensor Board Callback :

# In[39]:


tb_callback = TensorBoard(log_dir=results_path / 'logs',
                          histogram_freq = 5,
                          write_graph = True,
                          write_images = True)


# ### Checkpoint Callback :

# In[40]:


filepath = (results_path / 'autencoder.32.weights.hdf5').as_posix()


# In[42]:


checkpointer = ModelCheckpoint(filepath = filepath, 
                               monitor = 'val_loss', 
                               save_best_only = True,
                               save_weights_only = True,
                               mode = 'auto')


# ### Training The Model :

# In[43]:


training = autoencoder.fit(x = X_train_scaled,
                           y = X_train_scaled,
                           epochs = 100,
                           batch_size = 32,
                           shuffle = True,
                           validation_split = .1,
                           callbacks=[tb_callback, early_stopping, checkpointer])


# In[44]:


autoencoder.load_weights(filepath)


# ### Evaluate Trained Model :

# In[45]:


mse = autoencoder.evaluate(x=X_test_scaled, y=X_test_scaled)
f'MSE: {mse:.4f} | RMSE {mse**.5:.4f}'


# ### Encoding And Decoding Test Images :

# To encode data, we use the encoder we just defined:

# In[46]:


encoded_test_img = encoder.predict(X_test_scaled)
encoded_test_img.shape


# Decoder takes the compressed data and reproduces the output according to the autoencoder training results:

# In[47]:


decoded_test_img = decoder.predict(encoded_test_img)
decoded_test_img.shape


# ### Compare Original With Reconstructed Images :

# The following figure shows ten original images and their reconstruction by the autoencoder and illustrates the loss after compression:

# In[48]:


fig, axes = plt.subplots(ncols=n_classes, nrows=2, figsize=(20, 4))
for i in range(n_classes):
    
    axes[0, i].imshow(X_test_scaled[i].reshape(image_size, image_size), cmap='gray')
    axes[0, i].axis('off')

    axes[1, i].imshow(decoded_test_img[i].reshape(28, 28) , cmap='gray')
    axes[1, i].axis('off')

fig.suptitle('Original and Reconstructed Images', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=.85)
fig.savefig(results_path / 'reconstructed', dpi=300)


# In[ ]:




