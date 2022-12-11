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

# + [markdown] colab_type="text" id="aKcq7_ZqB7LI" pycharm={"name": "#%% md\n"}
# # Hey Everyone! Here's my progress in learning Tensorflow and practicing Machine Learning. **Part 1**
#
#
# **Follow my journey on social media:** [Podcast](https://open.spotify.com/show/6FxUBKO4bqwRWsjAIGZMwz) | [Twitter](https://twitter.com/tlkdata2me) | [Instagram](https://www.instagram.com/tlkdata2me/) | [LinkedIn](https://www.linkedin.com/in/shecananalyze/) 
#
# Learning Source: [Click here to take freeCodeCamp's TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk)

# + [markdown] colab_type="text" id="oixCgcYYH5bb" pycharm={"name": "#%% md\n"}
# # Intro to Tensorflow
# Vector - A **tensor** is a **vector** of a number (n) of dimensions that represent all types of data which are represented in *shapes.*
#
# Shapes - The dimension of the data being represented. (EX 2 rows 2 Columns)
#
# **Graphs and Sessions**
# - Graph - Set of computations that take place one after the other
# - Session - At this point I believe sessions are components from the graph (I will update this upon understanding)
#
# **Types of Tensors**
# - Variable - Right now I understand that variables can't be changed. (Immutable) 

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="Ax4VjKJGE7DO" outputId="98823470-dc8d-408f-90ea-3c8d165ffc2b" pycharm={"name": "#%%\n"}
import tensorflow as tf
print(tf.version)

# + colab={} colab_type="code" id="iwGmGpEQMipe" pycharm={"name": "#%%\n"}
# Video Example (We Do)
#Here I created a tensor of 0's with the shape of 5 sets of 5 arrays 5 rows with 5 columns
t = tf.zeros([5,5,5,5]) 
#print(t)
#Reshapes the tensor with 625 elements (elements are the number 0 in this case)
t = tf.reshape(t,[625]) 
#print(t)
#Reshapes the tensor with 125 rows (using -1 will create the amoun of columns needed to complete the task)
t = tf.reshape(t,[125, -1]) 
#print(t)

# + colab={} colab_type="code" id="IFYJPXPBNKdl" pycharm={"name": "#%%\n"}
# My Check on Learning (I Do)
c = tf.zeros([2, 3, 4, 5])
#print(c)
c = tf.reshape(c, [5, 4, 3, 2])
#print(c)
# Or
c = tf.reshape(c, [15, -1])
#print(c)

# + [markdown] colab_type="text" id="hRbfSFExURZM" pycharm={"name": "#%% md\n"}
# # Core Learning Algoriths

# + [markdown] colab_type="text" id="XrOHRGILUcAZ" pycharm={"name": "#%% md\n"}
# ##  Linear Regression
# - Linear Regression - When data points are related linearly, we can use the given points to create a line of best fit and predict future values.

# + colab={} colab_type="code" id="7-7uU3waYDIP" pycharm={"name": "#%%\n"}
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from IPython.display import clear_output, display
from six.moves import urllib

import tensorflow as tf 

# + [markdown] colab_type="text" id="2cHkJQppfwrx" pycharm={"name": "#%% md\n"}
# Using the Titanic data set from Kaggle to predict the survival of passengers on the Titanic
#
# Steps to take when building a model
#
# 1. Load the data
# 2. Explore the data
# 3. Catagorize the data
# 4. Create feature columns for the data
#
#
#

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="WLN8k7dRYlTV" outputId="8ca0ed40-348e-4ac3-9782-846f16609e32" pycharm={"name": "#%%\n"}
# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
display(dftrain.head())
y_train = dftrain.pop('survived') #pop removes a column of information and saves it for later under the given variable name
y_eval = dfeval.pop('survived')
display(dftrain.head())

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="NvKHmQEsbDU7" outputId="e2df923f-0983-46f0-fb03-646fe7a0ef16" pycharm={"name": "#%%\n"}
#Explore the data
dftrain.describe() #Gives overall information on data set

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="4dVwpY6EcTm8" outputId="72ddc2f0-9fcd-4886-ed65-bfcdbb5e861c" pycharm={"name": "#%%\n"}
dftrain.shape #Shape of data

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="TkA-dBIPctp8" outputId="61bddeaf-7a26-41c5-9223-4e330174e507" pycharm={"name": "#%%\n"}
dftrain.age.hist(bins=20) #bins-increments of graphing

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="Bj8dtRtHdBhB" outputId="05378d9e-e76b-41ba-c81b-0b0d72d377f4" pycharm={"name": "#%%\n"}
dftrain.sex.value_counts().plot(kind='barh')

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="HfcT-YsEdU-B" outputId="da31872a-eb68-4d94-97b5-bb82be8bdeff" pycharm={"name": "#%%\n"}
dftrain['class'].value_counts().plot(kind='barh')

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="kFfwi2mAdZCf" outputId="dcc62094-f79f-4e1f-d3b2-27e2a98dca79" pycharm={"name": "#%%\n"}
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# + [markdown] colab_type="text" id="zVyZQf4Seitd" pycharm={"name": "#%% md\n"}
# - Most passengers are in their 20's or 30's
# - Most passengers are male
# - Most passengers are in "Third" class
# - Females have a much higher chance of survival

# + [markdown] pycharm={"name": "#%% md\n"}
# ##  Training vs Testing

# + [markdown] colab_type="text" id="cYGmfiw2e1aj" pycharm={"name": "#%% md\n"}
# **Training data** is what we feed to the model so that it can develop and learn.
#
# **Testing data** is what we use to evaulate the model and see how well it is performing. 

# + colab={} colab_type="code" id="pwzKVjxzdcD4" pycharm={"name": "#%%\n"}
# Categorize the data

# + [markdown] colab_type="text" id="ywzanI1Sgf7k" pycharm={"name": "#%% md\n"}
# **Categorical data** - non-numerical data that can be placed under a specified field
# - Note: Categorical data can be represented by numbers to identify the categories they belong to. (ex: Male = 1, Female = 0)
#
# **Numerical data** - data that's represented by numbers

# + colab={} colab_type="code" id="jF87tkcwe-mw" pycharm={"name": "#%%\n"}
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="_orfKwlhh0Mn" outputId="e978ab8d-5be9-4ac1-abd6-ff262d53cba7" pycharm={"name": "#%%\n"}
dftrain['class'].unique() #.unique() gives all of the unique values withing the dataset (the names of the different data elements)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="nesKdX_MhfWm" outputId="145a0597-a6ab-4b46-acb5-c296a6c11ea7" pycharm={"name": "#%%\n"}
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  #Create a column of feature names with the different associated vocabulary terms
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


# + [markdown] colab_type="text" id="dyDLCCiM5MP0" pycharm={"name": "#%% md\n"}
# ### Training process
# *  Models must be fed the data in batches
# * Batches are fed according to Epochs 
#   * Epochs are how many times the model will see the same data. Feeding the data to the model in variations
#   Note: Over feeding the model can harm the outcome so feed it a little at a time.
#   

# + [markdown] colab_type="text" id="GhNNjHM75s6y" pycharm={"name": "#%% md\n"}
# ### Input Function for Linear Regression
# The TensorFlow model we are going to use requires that the data we pass it comes in as a tf.data.Dataset object. This means we must create a **input function** that can **convert** our current **pandas dataframe into that object.**

# + colab={} colab_type="code" id="Gbf4QkyQjUOr" pycharm={"name": "#%%\n"}
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False) #num_epochs is 1 because we aren't training this dataset like the line above. Shuffle is False because we don't need to shuffle since we are testing it.

# + [markdown] colab_type="text" id="gHXKA0Uh6Qq6" pycharm={"name": "#%% md\n"}
# ### Creating the Linear Regression Model
#
# Here we will use a **linear estimator** utilize the linear regression algorithm.
#

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="U11rchVf51L3" outputId="700b4ec7-0159-4c04-8a3b-485e0ea36acd" pycharm={"name": "#%%\n"}
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier through an estimator module

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing data

clear_output()  # clears console output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model that tells the accuracy of it
#Accuracy - compares the dataset results with models predicted results to get the accuracy of the data
print(result)

# + [markdown] colab_type="text" id="fyXwWCwp7yHY" pycharm={"name": "#%% md\n"}
# ### Predicting the data set with the model
#
# How to make predictions for every point in the evaluation data set
#
#
#

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="Q2UcXp5b7QqX" outputId="ca85a1ea-c75c-4ace-95f2-1cc1fbbeb404" pycharm={"name": "#%%\n"}
#Check the predictions of the model
#Here we will turn the results into a list to get a dictionary of all points and predictions
result = list(linear_est.predict(eval_input_fn))
#print(result[0])
#Here are looking for the 'probabilities' dict because it will help us to see the probability that someone will survive or won't survive
#Look for this --> {'probabilities': array([0.9108078 (<--won't survive(0)) , 0.08919217 (<-- will survive(1))]}

#Here we will print the probability of survival (1)
print(result[0]['probabilities'][0]) #Format: [passenger/data point][data set dict][outcome (survival)]

# + [markdown] colab_type="text" id="hkqMOJpV1W8E" pycharm={"name": "#%% md\n"}
# To put this together we can evaluate the passengers' attributes to see if the prediction makes sense with the dfeval.loc[ ] method  

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="6mjZmP1lxkiM" outputId="3a13b463-9c77-4088-b60e-0c4a40f903c6" pycharm={"name": "#%%\n"}
print(dfeval.loc[36])
print(y_eval.loc[36])
print(result[36]['probabilities'][1]) 
#This will give us the passengers details and their chance of survival and if they survived

# + [markdown] colab_type="text" id="KMZ9SRQE38te" pycharm={"name": "#%% md\n"}
# ##  Classification
# Differentiating data points and separating them into classes. Predicting the probability that the data point is in specified classes. 

# + [markdown] colab_type="text" id="op51fZxt5Sur" pycharm={"name": "#%% md\n"}
# ### Dataset
# This specific dataset seperates flowers into 3 different classes of species.
#
# * Setosa
# * Versicolor
# * Virginica
#
# The information about each flower is the following.
#
# * sepal length
# * sepal width
# * petal length
# * petal width

# + colab={} colab_type="code" id="gkl1Ccip5R3z" pycharm={"name": "#%%\n"}
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

# + colab={"base_uri": "https://localhost:8080/", "height": 107} colab_type="code" id="arIFAw7U2Hcf" outputId="cc66e61e-0ad1-4a50-91ce-23650b4cd350" pycharm={"name": "#%%\n"}
#Call in the data set
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

# + colab={"base_uri": "https://localhost:8080/", "height": 197} colab_type="code" id="eemrOkOl6DRP" outputId="d429c7fe-8f21-4f96-a047-f500bcbe14c3" pycharm={"name": "#%%\n"}
train.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 197} colab_type="code" id="_lU-guKU6QRC" outputId="fdfb7039-d022-4fdb-877f-993eeddce141" pycharm={"name": "#%%\n"}
train_y = train.pop('Species')
test_y = test.pop('Species')
train.head() # the species column is now gone

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="uPYKsNu66mCT" outputId="2213bd4b-d669-4d09-8c2b-48970e645817" pycharm={"name": "#%%\n"}
train.shape


# + [markdown] colab_type="text" id="GGTDLr_t7uHd" pycharm={"name": "#%% md\n"}
# ### Input Function for Classification

# + colab={"base_uri": "https://localhost:8080/", "height": 132} colab_type="code" id="q_vQWLXY6rRA" outputId="13935eb7-1ae5-46b1-f28a-398075e49b10" pycharm={"name": "#%%\n"}
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffling and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Feature columns describe how to use the input. We don't need the numerical of the previous input function because we're looking for the key values
my_feature_columns = []
#this code will loop through all of the keys in the dataset
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# + [markdown] colab_type="text" id="Ag1dTns48uMc" pycharm={"name": "#%% md\n"}
# ##  Building the Classification Model
#
# Here we will use the DNN Classifier (Deep Neural Network)

# + colab={} colab_type="code" id="c9SDMlkf8J8z" pycharm={"name": "#%%\n"}
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# + colab={} colab_type="code" id="ZTW8OsBW-Koh" pycharm={"name": "#%%\n"}
#Here we trained the model
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
# We include a lambda to avoid creating an inner function previously

# + colab={} colab_type="code" id="omHaMKYJ_JyD" pycharm={"name": "#%%\n"}
#Here we will evaluate the model
#We didn't specify the number of steps because during evaluation the model will only look at the testing data one time.
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# + [markdown] colab_type="text" id="pKBuojugA3cI" pycharm={"name": "#%% md\n"}
#

# + colab={} colab_type="code" id="uFTAoOYFActO" pycharm={"name": "#%%\n"}
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))

# + colab={} colab_type="code" id="CWiHaZIlCq2H" pycharm={"name": "#%%\n"}
print(pred_dict)

#Probability percentage tells you how likely the data point is to be classified as the key value predictions.
#Class ID tells you what value the prediction is from the original list.

# + [markdown] colab_type="text" id="RzP2QYFDA0DN" pycharm={"name": "#%% md\n"}
# ##  Clustering
#
# **Clustering** is an *unsupervised learning algorithm*. It finds clusters of like data and tells you the location of the clusters. 
#
# ### Basic Algorithm for K-Means.
# - Step 1: Randomly pick K points to place K centroids
#   - **Centroids** are the base of a cluster and tells you where the needed cluster is.
# - Step 2: Assign all the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
# - Step 3: Average all the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
# - Step 4: Reassign every point once again to the closest centroid.
# - Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.

# + [markdown] colab_type="text" id="-f47NKw5JFQ7" pycharm={"name": "#%% md\n"}
# ##  Hidden Markov Models
# "The Hidden Markov Model is a finite set of states, each of which is associated with a (generally multidimensional) probability distribution []. Transitions among the states are governed by a set of probabilities called transition probabilities." (http://jedlik.phy.bme.hu/~gerjanos/HMM/node4.html)
#
# A hidden markov model works with probabilities to predict future events or states. In this section we will learn how to create a hidden markov model that can predict the weather.
#
# This section is based on the following TensorFlow tutorial. https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel
#
# ### Data
# Let's start by discussing the type of data we use when we work with a hidden markov model.
#
# In the previous sections we worked with large datasets of 100's of different entries. For a markov model we are only interested in probability distributions that have to do with states.
#
# We can find these probabilities from large datasets or may already have these values. We'll run through an example in a second that should clear some things up, but let's discuss the components of a markov model.
#
# **States**: In each markov model we have a finite set of states. These states could be something like "warm" and "cold" or "high" and "low" or even "red", "green" and "blue". These states are "hidden" within the model, which means we do not direcly observe them.
#
# **Observations**: Each state has a particular outcome or observation associated with it based on a probability distribution. An example of this is the following: On a hot day Tim has a 80% chance of being happy and a 20% chance of being sad.
#
# **Transitions**: Each state will have a probability defining the likelyhood of transitioning to a different state. An example is the following: a cold day has a 30% chance of being followed by a hot day and a 70% chance of being follwed by another cold day.
#
# To create a hidden markov model we need.
#
# - States
# - Observation Distribution
# - Transition Distribution
#
#
# For our purpose we will assume we already have this information available as we attempt to predict the weather on a given day.

# + colab={} colab_type="code" id="71eqGzF8RWv8" pycharm={"name": "#%%\n"}
import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf
import numpy as np


# + [markdown] colab_type="text" id="cenp3DTieuEf" pycharm={"name": "#%% md\n"}
# Here we coded a model for the following:
#
# 1. Cold days are encoded by a 0 and hot days are encoded by a 1.
# 2. The first day in our sequence has an 80% chance of being cold.
# 3. A cold day has a 30% chance of being followed by a hot day.
# 4. A hot day has a 20% chance of being followed by a cold day.
# 5. On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.
#
# **Standard deviation** gives us the range of values above or below the mean
#

# + colab={} colab_type="code" id="M5wwoVekEHm5" pycharm={"name": "#%%\n"}
#Here we create probability of distribution model
tfd = tfp.distributions  # making a shortcut for later on
#bracket probability representation - [Cold Day, Hot Day]
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # insert standard deviation. Refer to point 5 above
#loc=[mean] scale=[standard deviation]
# the loc argument represents the mean and the scale is the standard devitation

# + colab={} colab_type="code" id="-6zMGrr5Rcye" pycharm={"name": "#%%\n"}
#Here we use hidden Markov model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

# + [markdown] colab_type="text" id="VDcskrMNTVel" pycharm={"name": "#%% md\n"}
# The number of steps represents the number of days that we would like to predict information for. In this case we've chosen 7, an entire week.
#
# To get the expected temperatures on each day we can do the following.

# + colab={} colab_type="code" id="X0zs4E-qTVGu" pycharm={"name": "#%%\n"}
mean = model.mean() #Partially defined tensor - This calculates the probablity so that we can run our probability distribution model

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())

# + [markdown] colab_type="text" id="e4z2zK5Sk3Wu" pycharm={"name": "#%% md\n"}
# The model and session combines gives the temperature on 7 days (the reason we input a 7 in the steps portion of the model
#
# Note: temperature is in celsius
#
# - [**Day 1:** 2.9999998 | **Day 2:** 5.9999995 | **Day 3:** 7.4999995 | **Day 4:** 8.25 | **Day 5:** 8.625001 | **Day 6:** 8.812501 | **Day 7:** 8.90625  ]
