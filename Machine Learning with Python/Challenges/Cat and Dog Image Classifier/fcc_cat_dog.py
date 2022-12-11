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

# + [markdown] id="XeCllzC77-P5" pycharm={"name": "#%% md\n"}
# *Note: You are currently reading this using Google Colaboratory which is a cloud-hosted version of Jupyter Notebook. This is a document containng both text cells for documentation and runnable code cells. If you are unfamiliar with Jupyter Notebook, watch this 3-minute introduction before starting this challenge: https://www.youtube.com/watch?v=inN8seMm7UI*
#
#
#
# ---
#
#
#
# For this challenge, you will complete the code below to classify images of dogs and cats. You will use Tensorflow 2.0 and Keras to create a convolutional neural network that correctly classifies images of cats and dogs at least 63% of the time. (Extra credit if you get it to 70% accuracy!)
#
# Some of the code is given to you but some code you must fill in to complete this challenge. Read the instruction in each text cell so you will know what you have to do in each code cell.
#
# The first code cell imports the required libraries. The second code cell downloads the data and sets key variables. The third cell is the first place you will write your own code.
#
# The structure of the dataset files that are downloaded looks like this (You will notice that the test directory has no subdirectories and the images are not labeled):
# ```
# cats_and_dogs
# |__ train:
#     |______ cats: [cat.0.jpg, cat.1.jpg ...]
#     |______ dogs: [dog.0.jpg, dog.1.jpg ...]
# |__ validation:
#     |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
#     |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
# |__ test: [1.jpg, 2.jpg ...]
# ```
#
# You can tweak epochs and batch size if you like, but it is not required.

# + id="la_Oz6oLlub6" pycharm={"name": "#%%\n"}
try:
  # This command only in Colab.
  # %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# + colab={"base_uri": "https://localhost:8080/"} id="jaF8r6aOl48C" outputId="16ca48e2-c09a-4c63-ab25-296c7e813f41" pycharm={"name": "#%%\n"}
# Get project files
# !wget -nc https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip

# !unzip -n cats_and_dogs.zip

PATH = 'cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# + [markdown] id="54bSbGlK9-6T" pycharm={"name": "#%% md\n"}
# Now it is your turn! Set each of the variables below correctly. (They should no longer equal `None`.)
#
# Create image generators for each of the three image data sets (train, validation, test). Use `ImageDataGenerator` to read / decode the images and convert them into floating point tensors. Use the `rescale` argument (and no other arguments for now) to rescale the tensors from values between 0 and 255 to values between 0 and 1.
#
# For the `*_data_gen` variables, use the `flow_from_directory` method. Pass in the batch size, directory, target size (`(IMG_HEIGHT, IMG_WIDTH)`), class mode, and anything else required. `test_data_gen` will be the trickiest one. For `test_data_gen`, make sure to pass in `shuffle=False` to the `flow_from_directory` method. This will make sure the final predictions stay is in the order that our test expects. For `test_data_gen` it will also be helpful to observe the directory structure.
#
#
# After you run the code, the output should look like this:
# ```
# Found 2000 images belonging to 2 classes.
# Found 1000 images belonging to 2 classes.
# Found 50 images belonging to 1 classes.
# ```

# + colab={"base_uri": "https://localhost:8080/"} id="EOJFeEfumns6" outputId="574d3c8d-bcb9-4a9d-8cb4-61fa1d6dfc11" pycharm={"name": "#%%\n"}
train_image_generator = ImageDataGenerator(rescale=1/255)
validation_image_generator = ImageDataGenerator(rescale=1/255)
test_image_generator = ImageDataGenerator(rescale=1/255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
    )

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
    )

test_data_gen = test_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=["."],
    class_mode="binary",
    shuffle=False
    )


# + [markdown] id="77TvZVETP_b2" pycharm={"name": "#%% md\n"}
# The `plotImages` function will be used a few times to plot images. It takes an array of images and a probabilities list, although the probabilities list is optional. This code is given to you. If you created the `train_data_gen` variable correctly, then running the cell below will plot five random training images.

# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="TP0WA8j1mt7Q" outputId="28955455-74eb-40ea-db95-8d81c1aba990" pycharm={"name": "#%%\n"}
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])


# + [markdown] id="GlLF4j7hkxEp" pycharm={"name": "#%% md\n"}
# Recreate the `train_image_generator` using `ImageDataGenerator`. 
#
# Since there are a small number of training examples there is a risk of overfitting. One way to fix this problem is by creating more training data from existing training examples by using random transformations.
#
# Add 4-6 random transformations as arguments to `ImageDataGenerator`. Make sure to rescale the same as before.
#

# + id="-32RRLY_3voj" pycharm={"name": "#%%\n"}
train_image_generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split = 0.2,
    rescale=1/255,
    fill_mode="nearest"
)

# + [markdown] id="JLzS54VmqkRg" pycharm={"name": "#%% md\n"}
# You don't have to do anything for the next cell. `train_data_gen` is created just like before but with the new `train_image_generator`. Then, a single image is plotted five different times using different variations.

# + colab={"base_uri": "https://localhost:8080/", "height": 863} id="pkwq2LFvqabS" outputId="26a2e5a1-1c2d-46c3-8460-e88777efae10" pycharm={"name": "#%%\n"}
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

# + [markdown] id="wUPWeKtESzA9" pycharm={"name": "#%% md\n"}
# In the cell below, create a model for the neural network that outputs class probabilities. It should use the Keras Sequential model. It will probably involve a stack of Conv2D and MaxPooling2D layers and then a fully connected layer on top that is activated by a ReLU activation function.
#
# Compile the model passing the arguments to set the optimizer and loss. Also pass in `metrics=['accuracy']` to view training and validation accuracy for each training epoch.

# + colab={"base_uri": "https://localhost:8080/"} id="k8aZkwMam4UY" outputId="ac75043c-3a7f-462e-ff50-50faa9e0e7f7" pycharm={"name": "#%%\n"}
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2))

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.summary()

# + [markdown] id="-rYmKuT1dRdy" pycharm={"name": "#%% md\n"}
# Use the `fit` method on your `model` to train the network. Make sure to pass in arguments for `x`, `steps_per_epoch`, `epochs`, `validation_data`, and `validation_steps`.

# + colab={"base_uri": "https://localhost:8080/"} id="1niQDz5x6K7y" outputId="9bc3a359-2756-4401-9c53-0dde11182d80" pycharm={"name": "#%%\n"}
history = model.fit(
    x=train_data_gen,
    validation_data=val_data_gen,
    epochs=epochs,
    steps_per_epoch=total_train // batch_size,
    validation_steps=total_val // batch_size
)

# + [markdown] id="Imux8tZWsWp-" pycharm={"name": "#%% md\n"}
# Run the next cell to visualize the accuracy and loss of the model.

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="5xS51mB56OAC" outputId="10931b75-5901-409b-f9e9-a366f55f2a35" pycharm={"name": "#%%\n"}
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# + [markdown] id="2bWrqi-usz3y" pycharm={"name": "#%% md\n"}
# Now it is time to use your model to predict whether a brand new image is a cat or a dog.
#
# In this final cell, get the probability that each test image (from `test_data_gen`) is a dog or a cat. `probabilities` should be a list of integers. 
#
# Call the `plotImages` function and pass in the test images and the probabilities corresponding to each test image.
#
# After your run the cell, you should see all 50 test images with a label showing the percentage sure that the image is a cat or a dog. The accuracy will correspond to the accuracy shown in the graph above (after running the previous cell). More training images could lead to a higher accuracy.

# + id="vYrSifOit2aK" pycharm={"name": "#%%\n"}
predictions = model.predict(test_data_gen)
probabilities = np.argmax(model.predict(test_data_gen), axis=-1)

# + [markdown] id="222ZZUup6SVO" pycharm={"name": "#%% md\n"}
# Run this final cell to see if you passed the challenge or if you need to keep trying.

# + colab={"base_uri": "https://localhost:8080/"} id="4IH86Ux_u7TZ" outputId="4f173e3e-b7b1-4516-dfc0-8703886015b0" pycharm={"name": "#%%\n"}
answers =  [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 
            0, 0, 0, 0, 0, 0]

correct = 0

for probability, answer in zip(probabilities, answers):
  if round(probability) == answer:
    correct +=1

percentage_identified = (correct / len(answers))

passed_challenge = percentage_identified > 0.63

print(f"Your model correctly identified {round(percentage_identified, 2)}% of the images of cats and dogs.")

if passed_challenge:
  print("You passed the challenge!")
else:
  print("You haven't passed yet. Your model should identify at least 63% of the images. Keep trying. You will get it!")
