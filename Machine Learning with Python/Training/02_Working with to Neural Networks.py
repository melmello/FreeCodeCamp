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

# + [markdown] colab_type="text" id="aKcq7_ZqB7LI"
# # Hey Everyone! Here's my progress in learning Tensorflow and practicing Machine Learning. **Part 2**
#
#
# **Follow my journey on social media:** [Podcast](https://open.spotify.com/show/6FxUBKO4bqwRWsjAIGZMwz) | [Twitter](https://twitter.com/tlkdata2me) | [Instagram](https://www.instagram.com/tlkdata2me/) | [LinkedIn](https://www.linkedin.com/in/shecananalyze/) 
#
# Learning Source: [Click here to take freeCodeCamp's TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk)

# + [markdown] colab_type="text" id="hjyKEyET_oWC"
# ## What I Learned about Neural Networks (Summary)
# So I listened to Tim explain Neural Networks and how they work. I will not place detailed notes in the notebook, but I have placed the link to his course above. 
#
# The bases of Neural Networks is:
# - Neural Network include
#   - Inputs
#   - Outputs
#   - Hidden Layers
#     - Layers are connected by weights
#     - There are Biases on each layer
#       - Biases are controlled neurons placed in each layer before the output and and allows us to move the network in the needed direction. 
#       
# Calculating the Network: 
# 1. Take the weighted sum of each Neuron in the layer by adding together all of the weights that connect the Neurons from the previous layer to the Neuron we need to find the value for.(This is done for each neuron in question)
# 2. Add the Bias
# 3. Apply Activation Functions to Biases to move them according to the function and help place neurons in the hidden layers between two values to determine the output. 
#   - Activation Function moves the network up in complexity
# 4. This process is repeated until the last output layer
#
# Training the Network:
# 1. Make predictions
# 2. Compare predictions to the expected values calculated in the Loss/Cost functions 
#   - The lower the loss function, the better the network
# 3. Calculate the Gradient
# 4. Use the back-propigation algorithm to back track through the neural network and adjust weights and biases according to the calculated gradient.

# + [markdown] colab_type="text" id="kDQFq6UzcuHh"
# ### Optimizer
#
# **Optimizer (optimization function)** - A function that implements the backpropagation algorithm. 
#
# A list of a few common optimizers:
#
# - Gradient Descent
# - Stochastic Gradient Descent
# - Mini-Batch Gradient Descent
# - Momentum
# - Nesterov Accelerated Gradient

# + [markdown] colab_type="text" id="yD_Ga7sudUvV"
# ## Creating a Neural Network

# + colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="2UX1KP-H_gsE" outputId="52c4f799-cb44-42bb-e6d4-2b51adee95d6"
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# + [markdown] colab_type="text" id="bCHgDPm3fMsA"
# Here we imported a dataset from Keras which included 60,000 images for training and 10,000 images for validation/testing.
#

# + colab={} colab_type="code" id="06OyGTWbd4az"
fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training sets

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="oTcGnU5neWhT" outputId="7c562018-2f43-4e4c-de0b-79503b6c0576"
#Sets are train_images, train_labels, test_images, and test_labels
train_images.shape

# + [markdown] colab_type="text" id="GGux21tUexSb"
# This shape means
# (60000 *images*, 28 *pixels*, 28 *pixels*)
#
# So we've got 60,000 images that are made up of 28x28 pixels (784 pixels in total)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="qn7XwbVMesme" outputId="390e97df-37ee-4639-fa8d-3d419338a9b2"
train_images[0,23,23]  # let's have a look at one pixel

# + [markdown] colab_type="text" id="DvHLabQH3hjJ"
# From the train_images set we are calling: [image number 0, row 23(pixel), column 23(pixel)] 
#
# Our pixel values are between 0 and 255, 0 being black and 255 being white. This means we have a grayscale image as there are no color channels.

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="AMJvAscM3lF1" outputId="5955b1c0-9525-41b8-9413-b549be55f6fc"
train_labels[:10]  # let's have a look at the first 10 training labels

# + [markdown] colab_type="text" id="y6QsaSYP36nZ"
# Our labels are integers ranging from 0 - 9. Each integer represents a specific article of clothing. We'll create an array of label names to indicate which is which.

# + colab={} colab_type="code" id="0u0L1QyI330R"
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# + [markdown] colab_type="text" id="ipBqfa-d4NbZ"
# Seeing what the images look like

# + colab={"base_uri": "https://localhost:8080/", "height": 265} colab_type="code" id="4eDd9HbA4HwR" outputId="d4f545c8-2647-4a4c-bf5d-edc524708195"
#This block of code will use madplotlib to help you see a grid view of the pixelated image. 
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# + [markdown] colab_type="text" id="0n67yX4483_n"
# ## Data Preprocessing
#
# The last step before creating our model is to *preprocess* our data. This simply means applying some prior transformations to our data before feeding it the model. In this case we will simply scale all our greyscale pixel values (0-255) to be between 0 and 1. We can do this by dividing each value in the training and testing sets by 255.0. We do this because smaller values will make it easier for the model to process our values. 
#

# + colab={} colab_type="code" id="5BtB8vzJ8V9Y"
train_images = train_images / 255.0

test_images = test_images / 255.0
#I will have to practice this step to full understand and get some good practice with it. 

# + [markdown] colab_type="text" id="-HCZ80CY9SHY"
# ## Building the Model
# Now it's time to build the model! We are going to use a keras *sequential* model with three different layers. This model represents a feed-forward neural network (one that passes values from left to right). We'll break down each layer and its architecture below.
#
# *He put pretty great notes here so most of it will be copied from the original doc.*

# + colab={} colab_type="code" id="-OkWlfud9KCX"
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

# + [markdown] colab_type="text" id="U0p5321U9zdK"
# **Sequential Neural Networks:** Calculate from the left side to the right side of the neural network (in order)
#
# **Layer 1:** This is our input layer and it will conist of 784 neurons. We use the flatten layer with an input shape of (28,28) to denote that our input should come in in that shape. The ***.Flatten*** means that our layer will reshape the shape (28,28) array into a vector of 784 neurons so that each pixel will be associated with one neuron.
#
# **Layer 2:** This is our first and only hidden layer. The ***.Dense*** denotes that this layer will be fully connected and each neuron from the previous layer connects to each neuron of this layer. It has 128 neurons and uses the rectify linear unit *'relu'* activation function coded as *(activation = 'relu')*.
#
# **Layer 3:** This is our output later and is also a dense layer. **The number of output neurons (10) should be the same as the number of classes (10).** It has 10 neurons that we will look at to determine our models output. Each neuron represnts the probabillity of a given image being one of the 10 different classes. The activation function *softmax* is used on this layer to calculate a probabillity distribution for each class. This means the value of any neuron in this layer will be between 0 and 1, where 1 represents a high probabillity of the image being that class.

# + [markdown] colab_type="text" id="kWbvUJR_BCJG"
# ### Compile the Model
# **The last step in building the model is to define the following**
# - optimizer
# - loss function 
# - metrics we would like to track 
#
# **Hyperparameter:** a parameter whose value is set before the learning process begins. Another way to put it is that they are settings that can be tuned to control the behavior of a machine learning algorithm.
#
# **Hyperparameter Tuning/Optimizations:** Choosing a set of hyperparameters for a learning algorithm

# + colab={} colab_type="code" id="PRGezbkdAyYI"
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# + [markdown] colab_type="text" id="Xuqmxym7C0PP"
# ## Training the Model
# Now it's finally time to train the model. Since we've already done all the work on our data this step is as easy as calling a single method.

# + colab={"base_uri": "https://localhost:8080/", "height": 222} colab_type="code" id="yoU8x50GCrwe" outputId="c0b94894-34d2-465b-a240-eaf9cac5f83b"
model.fit(train_images, train_labels, epochs=5)  # we pass the data, labels and epochs and watch the magic!

# + [markdown] colab_type="text" id="ekOdUfXdFCiz"
# ## Evaluating the Model
# Now it's time to test/evaluate the model. We can do this quite easily using another builtin method from keras.
#
# The *verbose* argument is defined from the keras documentation as:
# "verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar."
# (https://keras.io/models/sequential/)

# + colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="GxykOXLkFAvl" outputId="25e0a8ae-9806-4c39-9218-566c48dc97b4"
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

print('Test accuracy:', test_acc)

# + [markdown] colab_type="text" id="T26X6H3JGGF-"
# **Overfitting** is when the accuracy here is lower than when training the model. 
#
# The model is now trained and we can use it to predict values. 

# + [markdown] colab_type="text" id="9txSQaDAJ_By"
# ## Making Predictions
# To make predictions we simply need to pass an array of data in the form we've specified in the input layer to ```.predict()``` method.

# + colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="7GmdmT41KDgo" outputId="f6722139-cf65-4a19-b781-e4b2cd02a7c3"
predictions = model.predict(test_images)
print(predictions[0])

# + [markdown] colab_type="text" id="uW4FTywiKIiB"
# This method returns to us an array of predictions (probability distribution) for each image. Each number represents a class from class_names, the larger number represents the class the model is predicting the array is an image of. 

# + colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="uuoDbeLnGFqp" outputId="f0cf1e63-d13f-41c6-d5c9-725d6c9f953b"
predictions[0]

# + [markdown] colab_type="text" id="ObPGWbbYKQj8"
# If we wan't to get the value with the highest score we can use a useful function from numpy called **argmax**(). This simply returns the index of the maximium value from a numpy array.
#

# + [markdown] colab_type="text" id="C1izE_md4JHD"
#

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="m7eFv_buF44a" outputId="006b753f-1bed-4970-9999-835864fb712e"
np.argmax(predictions[0]) #The number in the 9th postion(really 10th) of the array is the maximum value

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="cXJWHqSsMZyK" outputId="5d83d228-d9a5-4bd7-9c5a-b2372a4d99ab"
print(class_names[np.argmax(predictions[0])]) #The label in the 9th position is the label the model is predicting is correct

# + [markdown] colab_type="text" id="4SuF22oqKV0Q"
# we can check if this is correct by looking at the value of the cooresponding test label.

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="CDGp1G1hKfXT" outputId="0d1cb56b-abd2-4154-cad2-368bf1169fd4"
test_labels[0]

# + [markdown] colab_type="text" id="dNl0M8tLNaNT"
# Putting it all togther here because I like how it was done in the video. 

# + colab={"base_uri": "https://localhost:8080/", "height": 282} colab_type="code" id="lFEI1UEmNm4X" outputId="591216d2-5f33-4d75-c7c7-7eeab4cd6f36"
print('The model has guessed: {}'.format(class_names[np.argmax(predictions[19])])) #Enter prediction index
plt.figure()
plt.imshow(test_images[19]) #enter image index
plt.colorbar()
plt.grid(False)
plt.show()
#The model should return what it's prediction of the pixelated image is. 
#The matplotlib portion of the code block will return a graph of the image in question

# + [markdown] colab_type="text" id="V7O7pObXKjts"
# ## Verifying Predictions
#
# The model will have us pick a number, it will then tell us the expected image along with the grid and then tell us what the guess is....pretty much the same as above thing with an input option. Pretty cool!

# + colab={"base_uri": "https://localhost:8080/", "height": 312} colab_type="code" id="2nrmhM3fKkp9" outputId="6b90ee53-24a9-4e07-dcc7-da45c75ec8ce"
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)

