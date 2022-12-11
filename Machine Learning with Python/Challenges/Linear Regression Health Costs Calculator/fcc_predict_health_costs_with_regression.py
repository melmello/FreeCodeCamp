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

# + [markdown] id="M9TX15KOkPBV" pycharm={"name": "#%% md\n"}
# *Note: You are currently reading this using Google Colaboratory which is a cloud-hosted version of Jupyter Notebook. This is a document containing both text cells for documentation and runnable code cells. If you are unfamiliar with Jupyter Notebook, watch this 3-minute introduction before starting this challenge: https://www.youtube.com/watch?v=inN8seMm7UI*
#
# ---
#
# In this challenge, you will predict healthcare costs using a regression algorithm.
#
# You are given a dataset that contains information about different people including their healthcare costs. Use the data to predict healthcare costs based on new data.
#
# The first two cells of this notebook import libraries and the data.
#
# Make sure to convert categorical data to numbers. Use 80% of the data as the `train_dataset` and 20% of the data as the `test_dataset`.
#
# `pop` off the "expenses" column from these datasets to create new datasets called `train_labels` and `test_labels`. Use these labels when training your model.
#
# Create a model and train it with the `train_dataset`. Run the final cell in this notebook to check your model. The final cell will use the unseen `test_dataset` to check how well the model generalizes.
#
# To pass the challenge, `model.evaluate` must return a Mean Absolute Error of under 3500. This means it predicts health care costs correctly within $3500.
#
# The final cell will also predict expenses using the `test_dataset` and graph the results.

# + id="1rRo8oNqZ-Rj" pycharm={"name": "#%%\n"}
# Import libraries. You may or may not use all of these.
# !pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # # %tensorflow_version only exists in Colab.
  # %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from sklearn.model_selection import train_test_split

# + id="CiX2FI4gZtTt" pycharm={"name": "#%%\n"}
# Import data
# !wget -nc https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

# + id="LcopvQh3X-kX" pycharm={"name": "#%%\n"}
dataset["sex"] = pd.factorize(dataset["sex"])[0]
dataset["region"] = pd.factorize(dataset["region"])[0]
dataset["smoker"] = pd.factorize(dataset["smoker"])[0]

# + id="CBIlupN4ObK-" pycharm={"name": "#%%\n"}
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

# + id="XUttCFwtO4UQ" pycharm={"name": "#%%\n"}
train_labels = train_dataset.copy().pop('expenses')
test_labels = test_dataset.copy().pop('expenses')

# + id="HS9BVjVrO9sC" pycharm={"name": "#%%\n"}
normalizer = layers.experimental.preprocessing.Normalization()
normalizer.adapt(np.array(train_dataset))

model = keras.Sequential([
    normalizer,
    layers.Dense(16),
    layers.Dense(4),
    layers.Dropout(.2),
    layers.Dense(1),
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae', 'mse']
)

model.build()

# + id="CTAMzaTeT0mQ" pycharm={"name": "#%%\n"}
history = model.fit(
    train_dataset,
    train_labels,
    epochs=100,
    validation_split=0.2,
    verbose=0,
)

# + id="Xe7RXH3N3CWU" pycharm={"name": "#%%\n"}
# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)

