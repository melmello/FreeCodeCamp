# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] colab_type="text" id="cyrvI6jnVMRI" pycharm={"name": "#%% md\n"}
# # Hey Everyone! Here's my progress in learning Tensorflow and practicing Machine Learning. **Part 3**
#
#
# **Follow my journey on social media:** [Podcast](https://open.spotify.com/show/6FxUBKO4bqwRWsjAIGZMwz) | [Twitter](https://twitter.com/tlkdata2me) | [Instagram](https://www.instagram.com/tlkdata2me/) | [LinkedIn](https://www.linkedin.com/in/shecananalyze/) 
#
# Learning Source: [Click here to take freeCodeCamp's TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk)

# + [markdown] colab_type="text" id="5H88o6ZeVWIk" pycharm={"name": "#%% md\n"}
# #Deep Computer Vision 
# *(Copied from lesson notes)*
#
# In this guide we will learn how to peform *image classification and object detection/recognition* using deep computer vision with something called a **convolutional neural network**.
#
# The goal of our convolutional neural networks will be to classify and detect images or specific objects from within the image. We will be using image data as our features and a label for those images as our label or output.
#
# We already know how neural networks work so we can skip through the basics and move right into explaining the following concepts.
# - Image Data
# - Convolutional Layer
# - Pooling Layer
# - CNN Architectures
#
# The major differences we are about to see in these types of neural networks are the layers that make them up.

# + [markdown] colab_type="text" id="cQWC31DlWDi1" pycharm={"name": "#%% md\n"}
# ## Image Data
#
# Image data is usually made up of 3 dimensions. 
#
# These 3 dimensions are as follows:
# 1. image height
# 2. image width
# 3. color channels
#
# **Color Channels**
#
# The number of color channels represents the depth of an image and coorelates to the colors used in it. For example, an image with three channels is likely made up of rgb (red, green, blue) pixels. So, for each pixel we have three numeric values in the range 0-255 that define its color. For an image of color depth 1 we would likely have a greyscale image with one value defining each pixel, again in the range of 0-255.
#
# ![alt text](https://blog.xrds.acm.org/wp-content/uploads/2016/06/Figure1.png)

# + [markdown] colab_type="text" id="zppxqKIafneK" pycharm={"name": "#%% md\n"}
# ## Convolutional Neural Network
# **Dense Layer:** *A dense layer will consider the ENTIRE image in the exact pattern.* 
#
# **Convolutional (Neural Network) Layer:** *The convolutional layer will consider specific parts of the image anywhere in the photo.* 
#
# Each convolutional neural network is made up of one or many convolutional layers. These layers are different than the *dense* layers we have seen previously. Their goal is to find patterns from within images that can be used to classify the image or parts of it. But this may sound familiar to what our densly connected neural network in the previous section was doing, well that's becasue it is. 
#
# The fundemental difference between a dense layer and a convolutional layer is that dense layers detect patterns globally while convolutional layers detect patterns locally. When we have a densly connected layer each node in that layer sees all the data from the previous layer. This means that this layer is looking at all the information and is only capable of analyzing the data in a global capacity. Our convolutional layer however will not be densly connected, this means it can detect local patterns using part of the input data to that layer.
#
#

# + [markdown] colab_type="text" id="SCXaowYNh6yD" pycharm={"name": "#%% md\n"}
# ### How They Work
# A dense neural network learns patterns that are present in one specific area of an image. This means if a pattern that the network knows is present in a different area of the image it will have to learn the pattern again in that new area to be able to detect it. 
#
# *Let's use an example to better illustrate this.*
#
# We'll consider that we have a dense neural network that has learned what an eye looks like from a sample of dog images.
#
# ![alt text](https://drive.google.com/uc?export=view&id=16FJKkVS_lZToQOCOOy6ohUpspWgtoQ-c)
#
# Let's say it's determined that an image is likely to be a dog if an eye is present in the boxed off locations of the image above.
#
# Now let's flip the image.
# ![alt text](https://drive.google.com/uc?export=view&id=1V7Dh7BiaOvMq5Pm_jzpQfJTZcpPNmN0W)
#
# Since our **densly connected network** has only recognized patterns globally it will look where it thinks the eyes should be present. Clearly it does not find them there and therefore would likely determine this image is not a dog. Even though the pattern of the eyes is present, it's just in a different location.
#
# Since **convolutional layers** learn and detect patterns from different areas of the image, they don't have problems with the example we just illustrated. They know what an eye looks like and by analyzing different parts of the image can find where it is present. 

# + [markdown] colab_type="text" id="Q6kkJvo3iq7C" pycharm={"name": "#%% md\n"}
# ### Multiple Convolutional Layers
# (Will need to revisit)
# In our models it is quite common to have more than one convolutional layer. Even the basic example we will use in this guide will be made up of 3 convolutional layers. These layers work together by increasing complexity and abstraction at each subsequent layer. The first layer might be responsible for picking up edges and short lines, while the second layer will take as input these lines and start forming shapes or polygons. Finally, the last layer might take these shapes and determine which combiantions make up a specific image.
#
# ## Feature Maps
# **Feature map:** A 3D tensor with two spacial axes (width and height) and one depth axis. Our convolutional layers take feature maps as their input and return a **new** feature map that represents the prescence of specific filters from the **previous** feature map. These are what we call *response maps*.

# + [markdown] colab_type="text" id="mI6hSU-iVCJY" pycharm={"name": "#%% md\n"}
# Convolutional Layers (Summary of explaination)
#
# The properties of a convolutional layer are:
# - Input size
# - Filters
#   - Filters are a pattern of pixels
#    - We look for filters in each convolutional layer
# - Sample size of filters
#
# The process goes as follows:
# - The sample size filter is compared to the actual image and calculated (matrix wise) 
#   - Each sample size within the layer is shifted (stride) and then calculated all over again to create an output feature map
# - Once all of the filters are calculated in the input layer, we then start the process all over again for the next layer. 
#   - Instead of using pixels we are using the calculated numbers (output feature map) to find combinations of features that exist in the image. 
#     -This helps us to find lines, curves, combinations of lines and curves, etc.
#
# **Padding:** Adding the appropriate number of rows and/or columns to your input data such that each pixel can be centered by the filter.
#   - Adds a border to help look at the edges of a photo. 
#
# **Stride:** How many rows/cols we will move the filter each time. 
#
# **Pooling:** Taking specific values from a sample of an output feature map to reduce the size of the feature map. 
#
# Pooling Values:
# - Min
# - Max
# - Average
#  
#

# + [markdown] colab_type="text" id="RbMKJ1lVblsY" pycharm={"name": "#%% md\n"}
# ### Tutorial with Imaga Dataset
# The problem we will consider here is classifying 10 different everyday objects. The dataset we will use is built into tensorflow and called the CIFAR Image Dataset. It contains 60,000 32x32 color images with 6000 images of each class.

# + colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="9-y6Pi1KVAjx" outputId="8bb948c2-a43c-4725-a24d-41fe6c0afc9b" pycharm={"name": "#%%\n"}
import tensorflow as tf

from keras import datasets, layers, models
import matplotlib.pyplot as plt

# + colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="5G840u_dUlaQ" outputId="1e422e67-73c2-4b64-85d3-0d00b099898d" pycharm={"name": "#%%\n"}
#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# + colab={"base_uri": "https://localhost:8080/", "height": 280} colab_type="code" id="YGEZxqOHb2NO" outputId="5b2ef5e6-a6cb-46cf-b66a-1870230bd74c" pycharm={"name": "#%%\n"}
# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

# + [markdown] colab_type="text" id="VKAOkVrRcf1s" pycharm={"name": "#%% md\n"}
# Build the **Convolutional Base** 
#
# Here we will extract the features from the layer

# + colab={} colab_type="code" id="Y0ayDLnTb-t3" pycharm={"name": "#%%\n"}
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# + [markdown] colab_type="text" id="6IWsRKk9cmDi" pycharm={"name": "#%% md\n"}
# **Layer 1**
#
# The input shape of our data will be 32, 32, 3 and we will process 32 filters of size 3x3 over our input data. We will also apply the **activation function** relu to the output of each convolution operation.
#
# Note: Syntax is (32 **number of filters**, (3, 3 **grid size**(3x3)))
#
# **Layer 2**
#
# This layer will perform the max pooling operation using 2x2 samples and a stride of 2.
#
# **Other Layers**
#
# The next set of layers do very similar things but take as input the feature map from the previous layer. They also increase the frequency of filters from 32 to 64. We can do this as our data shrinks in spacial dimensions as it passed through the layers, meaning we can afford (computationally) to add more depth.

# + colab={"base_uri": "https://localhost:8080/", "height": 324} colab_type="code" id="QK-OHYXadpjt" outputId="72ccc9b6-1c8a-4335-fd8a-3afc09d7fdde" pycharm={"name": "#%%\n"}
model.summary()  # let's have a look at our model so far

# + [markdown] colab_type="text" id="cmew8KfBeJui" pycharm={"name": "#%% md\n"}
# ### Adding Dense Layers
#
# Take the convolutional base (extracted features) and add a way to classify them. 

# + colab={} colab_type="code" id="1JzgTokzdtpL" pycharm={"name": "#%%\n"}
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# + [markdown] colab_type="text" id="KgpmeZgRe7Ju" pycharm={"name": "#%% md\n"}
# We can see that the flatten layer changes the shape of our data so that we can feed it to the 64-node dense layer, followed by the final output layer of 10 neurons (one for each class).

# + colab={"base_uri": "https://localhost:8080/", "height": 427} colab_type="code" id="girGs0iAe59A" outputId="d2147066-5a2b-4404-aa90-616ab9baacf1" pycharm={"name": "#%%\n"}
model.summary()

# + [markdown] colab_type="text" id="VxBBrS-YflOJ" pycharm={"name": "#%% md\n"}
# ### Training the model
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 396} colab_type="code" id="CAgqf1JwfSFJ" outputId="8d935c5f-47dd-4564-c2c9-877046eed45a" pycharm={"name": "#%%\n"}
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# + [markdown] colab_type="text" id="wxfBHsnPjfHf" pycharm={"name": "#%% md\n"}
# ## Evaluating the Model
# We can determine how well the model performed by looking at it's performance on the test data set. And run predictions as we did before. 

# + colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="WRSexNs1gbHL" outputId="a773e77b-f2dc-498e-8d45-e161eaa1537e" pycharm={"name": "#%%\n"}
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# + [markdown] colab_type="text" id="E-Uc5y2ej8uC" pycharm={"name": "#%% md\n"}
#

# + colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="MXnyJoPgjl_y" outputId="ada4581e-6c48-4d3c-c539-58bb42185c98" pycharm={"name": "#%%\n"}
predictions = model.predict(test_images)
print(predictions[3])

# + colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="Cu3D7x-HkPtI" outputId="af91f2c2-dc1d-4d66-8368-cb004a7525a2" pycharm={"name": "#%%\n"}
import numpy as np

def pred(n):
  print(class_names[np.argmax(predictions[n])])
  print(test_labels[n])

pred(9)  


# + colab={"base_uri": "https://localhost:8080/", "height": 283} colab_type="code" id="cjAwEt7Qk13A" outputId="dfc15c1e-9fb4-44c2-95ec-f83bbbddf27b" pycharm={"name": "#%%\n"}
#The model should return what it's prediction of the pixelated image is. 
#The matplotlib portion of the code block will return a graph of the image in question

def predpic(n):
  print('The model has guessed: {}'.format(class_names[np.argmax(predictions[n])])) #Enter prediction index
  plt.figure()
  plt.imshow(test_images[n]) #enter image index
  plt.colorbar()
  plt.grid(False)
  plt.show()

predpic(9)

# + [markdown] colab_type="text" id="ym_ZVO2sooEZ" pycharm={"name": "#%% md\n"}
# ### Working with Small Datasets
#
# A few techniques if we're working with small data sets and need to expand the data

# + [markdown] colab_type="text" id="UD0BfeHjoueu" pycharm={"name": "#%% md\n"}
# ### Data Augmentation
# **Data Augmentation:** Performing random transformations on our images so that our model can generalize better. These transformations can be things like compressions, rotations, stretches and even color changes. This will allow us to avoid overfitting and create a larger dataset from a smaller one we can use a technique called 
# - In other words, we can take one image and pass different versions (flipped, stretched, rotated etc) of it through the model and augment it multiple times. 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="9KIV-LoCk-_s" outputId="23e0c5e5-3b8d-4735-c035-86de7c02b799" pycharm={"name": "#%%\n"}
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# creates a data generator object that transforms images
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

# pick an image to transform
test_img = test_images[9]
img = image.img_to_array(test_img)  # convert image to numpy arry
img = img.reshape((1,) + img.shape)  # reshape image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 4 images
        break

plt.show()


# + [markdown] colab_type="text" id="AxbfvVIPo76w" pycharm={"name": "#%% md\n"}
# ### Pretrained Models
#
# Here we will use part of an existing model and then "fine tune" it for the use of our own data. 
#
# ### Fine Tuning
# Tweaking the final layers in our convolutional base to work better for our specific problem. This involves not touching or retraining the earlier layers in our convolutional base but only adjusting the final few layers.

# + [markdown] colab_type="text" id="_EzTOxvjrcB-" pycharm={"name": "#%% md\n"}
# ## Using a Pretrained Model
# In this section we will combine the tecniques we learned above and use a pretrained model and fine tuning to classify images of dogs and cats using a small dataset.

# + colab={} colab_type="code" id="Ydqg4tTro5cm" pycharm={"name": "#%%\n"}
#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

# + [markdown] colab_type="text" id="UeEWpZZGo9lp" pycharm={"name": "#%% md\n"}
# ### Dataset Example
# We will load the *cats_vs_dogs* dataset from the modoule tensorflow_datatsets.
#
# This dataset contains (image, label) pairs where images have different dimensions and 3 color channels.

# + colab={"base_uri": "https://localhost:8080/", "height": 139} colab_type="code" id="xGP0PJVlo_Na" outputId="d741b078-8395-4aa0-d87f-f3bf1bb482e2" pycharm={"name": "#%%\n"}
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# + colab={"base_uri": "https://localhost:8080/", "height": 545} colab_type="code" id="1nl0fegYo_7Z" outputId="689f8f6c-a204-42df-c281-3e8e53dfdb23" pycharm={"name": "#%%\n"}
get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

# + [markdown] colab_type="text" id="z4qKDIOUpC1Z" pycharm={"name": "#%% md\n"}
# ### Data Preprocessing
# Since the sizes of our images are all different, we need to convert them all to the same size. We can create a function that will compress the size of the image for us below.

# + colab={} colab_type="code" id="r24CuskkpDf7" pycharm={"name": "#%%\n"}
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32) #cast meanst to convert (to a float in this case)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


# + [markdown] colab_type="text" id="A6XPACcUub5W" pycharm={"name": "#%% md\n"}
# Now we can apply this function to all our images using ```.map()```.

# + colab={} colab_type="code" id="pFvbioBVufcD" pycharm={"name": "#%%\n"}
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# + colab={"base_uri": "https://localhost:8080/", "height": 599} colab_type="code" id="zyuY0pC4uly1" outputId="69df759b-68d3-44c6-cfd8-e57f24e09be7" pycharm={"name": "#%%\n"}
for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

# + colab={} colab_type="code" id="lnCZ6vjHup16" pycharm={"name": "#%%\n"}
#No clue what this means at this moment
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# + colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="W4zlARQQvHP1" outputId="1babe8c8-0f76-45db-9df3-ba0813713c4d" pycharm={"name": "#%%\n"}
#Looking at the shape of the new image
for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)

# + [markdown] colab_type="text" id="0DMbxN1puwLy" pycharm={"name": "#%% md\n"}
# ## Picking a Pretrained Model
# The model we are going to use as the convolutional base for our model is the **MobileNet V2** developed at Google. This model is trained on 1.4 million images and has 1000 different classes.
#
# We want to use this model but only its convolutional base. So, when we load in the model, we'll specify that we don't want to load the top (classification) layer. We'll tell the model what input shape to expect and to use the predetermined weights from *imagenet* (Googles dataset).
#

# + colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" id="6vKh1qQ-vQeY" outputId="7ac1535b-e9b1-4595-eb84-5425d35971e7" pycharm={"name": "#%%\n"}
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="a_cm5mV1vUn0" outputId="a0615627-934f-4d39-cf26-c71a66561097" pycharm={"name": "#%%\n"}
base_model.summary()

# + [markdown] colab_type="text" id="FZKPy3vlvVka" pycharm={"name": "#%% md\n"}
# At this point this base_model will simply output a shape (32, 5, 5, 1280) tensor (from the last row in the summary) that is a feature extraction from our original (1, 160, 160, 3) image. The 32 means that we have 32 layers of differnt filters/features.

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="2IqMI2yFvY6z" outputId="4c5e5b1e-090e-451b-896b-6a03574f568c" pycharm={"name": "#%%\n"}
for image, _ in train_batches.take(1): # He went a little fast will look further into this with practice. 
   pass

feature_batch = base_model(image)
print(feature_batch.shape)

# + [markdown] colab_type="text" id="puKWO7wKvX-m" pycharm={"name": "#%% md\n"}
# ### Freezing the Base
# The term **freezing** refers to disabling the training property of a layer. It simply means we won’t make any changes to the weights of any layers that are frozen during training. This is important as we don't want to change the convolutional base that already has learned weights.

# + colab={} colab_type="code" id="VnZjoND_vgfz" pycharm={"name": "#%%\n"}
base_model.trainable = False

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="cPEVyYsdviFz" outputId="6d3de076-3175-481c-ec52-164ec7ce125e" pycharm={"name": "#%%\n"}
base_model.summary()

# + [markdown] colab_type="text" id="YQVunEAGxERE" pycharm={"name": "#%% md\n"}
# ### Adding our Classifier
# Now that we have our base layer setup, we can add the classifier. Instead of flattening the feature map of the base layer we will use a global average pooling layer that will average the entire 5x5 area of each 2D feature map and return to us a single 1280 element vector per filter.  

# + colab={} colab_type="code" id="LaXyxqGjw8B1" pycharm={"name": "#%%\n"}
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# + [markdown] colab_type="text" id="xUH5tRmHxV7j" pycharm={"name": "#%% md\n"}
# Finally, we will add the predicition layer that will be a single dense neuron. We can do this because we only have two classes to predict for.

# + colab={} colab_type="code" id="W3c6cmi3xXih" pycharm={"name": "#%%\n"}
prediction_layer = keras.layers.Dense(1)

# + [markdown] colab_type="text" id="kEa8gjSPxYpC" pycharm={"name": "#%% md\n"}
# Now we will combine these layers together in a model.
#

# + colab={} colab_type="code" id="8SNUAtLTxc9h" pycharm={"name": "#%%\n"}
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# + colab={"base_uri": "https://localhost:8080/", "height": 256} colab_type="code" id="Sp2KZeZ2xeIT" outputId="c58b8886-d326-42b9-a7a0-7e4941da9101" pycharm={"name": "#%%\n"}
model.summary()

# + [markdown] colab_type="text" id="sw2tY4aYyALn" pycharm={"name": "#%% md\n"}
# We have only **1,281 Trainable Parameters** because we only have 1280 connections from the *global_average_pooling2d layer* to the *dense_2 layer* and 1 Bias

# + [markdown] colab_type="text" id="AnKHif4eyrG_" pycharm={"name": "#%% md\n"}
# ### Training the Model

# + colab={} colab_type="code" id="cdgycl84x_6b" pycharm={"name": "#%%\n"}
base_learning_rate = 0.0001 #How much are we allowed to modify the network, it's low to ensure we don't make any major changes.
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="3jebgr0vxzJn" outputId="af698346-3ed6-49aa-9aaf-a8a8f5430463" pycharm={"name": "#%%\n"}
# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# + colab={"base_uri": "https://localhost:8080/", "height": 156} colab_type="code" id="5_pquq3Ay2ah" outputId="c27838ff-5a1c-452c-c319-46c5a5a18977" pycharm={"name": "#%%\n"}
# Now we can train it on our images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

# + colab={} colab_type="code" id="T4P8R4jEy2L0" pycharm={"name": "#%%\n"}
model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')
#from here we can predict like we did for the previous modules
