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

# + [markdown] colab_type="text" id="sr6FTpIeUiPN" pycharm={"name": "#%% md\n"}
# # Hey Everyone! Here's my progress in learning Tensorflow and practicing Machine Learning. **Part 4**
#
#
# **Follow my journey on social media:** [Podcast](https://open.spotify.com/show/6FxUBKO4bqwRWsjAIGZMwz) | [Twitter](https://twitter.com/tlkdata2me) | [Instagram](https://www.instagram.com/tlkdata2me/) | [LinkedIn](https://www.linkedin.com/in/shecananalyze/) 
#
# Learning Source: [Click here to take freeCodeCamp's TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk)

# + [markdown] colab_type="text" id="fQCHlhhEZN4q" pycharm={"name": "#%% md\n"}
# # Natural Language Processing
#
# ##  Recurrent Neural Networks
#
# **Recurrent Neural Network** A neural network that is capable of processing sequential data such as text or characters.
#
# ### Bag of Words
# **Bag of Words** When each word in a sentence is encoded with an integer and thrown into a collection that does not maintain the order of the words but does keep track of the frequency.
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="ZyYzm95VUVmN" outputId="24e62c8d-04d0-4bba-c091-46c751094288" pycharm={"name": "#%%\n"}
vocab = {}  # maps word to integer representing it
word_encoding = 1
def bag_of_words(text):
  global word_encoding

  words = text.lower().split(" ")  # create a list of all of the words in the text, well assume there is no grammar in our text for this example
  bag = {}  # stores all of the encodings and their frequency

  for word in words:
    if word in vocab:
      encoding = vocab[word]  # get encoding from vocab
    else:
      vocab[word] = word_encoding
      encoding = word_encoding
      word_encoding += 1
    
    if encoding in bag:
      bag[encoding] += 1
    else:
      bag[encoding] = 1
  
  return bag

text = "this is a test to see if this test will work is is test a a"
bag = bag_of_words(text)
print(bag)
print(vocab)

# + [markdown] colab_type="text" id="KHsBXoyLfFy0" pycharm={"name": "#%% md\n"}
# This isn't really the way we would do this in practice, but I hope it gives you an idea of how bag of words works. Notice that we've lost the order in which words appear. In fact, let's look at how this encoding works for the two sentences we showed above.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 72} colab_type="code" id="RXaduAn4fFUs" outputId="347b6429-73db-4f5a-b61c-00ab39ed053f" pycharm={"name": "#%%\n"}
positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_bag = bag_of_words(positive_review)
neg_bag = bag_of_words(negative_review)

print("Positive:", pos_bag)
print("Negative:", neg_bag)

# + [markdown] colab_type="text" id="ZmLJ4Gv1fJn8" pycharm={"name": "#%% md\n"}
# We can see that even though these sentences have a very different meaning they are encoded exaclty the same way.

# + [markdown] colab_type="text" id="g4t0v_JkfQRM" pycharm={"name": "#%% md\n"}
# ### Integer Encoding
# The next technique we will look at is called **integer encoding**. This involves representing each word or character in a sentence as a unique integer and maintaining the order of these words. This should hopefully fix the problem we saw before were we lost the order of words.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="fhGiDyAkc84b" outputId="e5173840-f49a-43a6-c963-8ff0bdc81e8b" pycharm={"name": "#%%\n"}
vocab = {}  
word_encoding = 1
def one_hot_encoding(text):
  global word_encoding

  words = text.lower().split(" ") 
  encoding = []  

  for word in words:
    if word in vocab:
      code = vocab[word]  
      encoding.append(code) 
    else:
      vocab[word] = word_encoding
      encoding.append(word_encoding)
      word_encoding += 1
  
  return encoding

text = "this is a test to see if this test will work is is test a a"
encoding = one_hot_encoding(text)
print(encoding)
print(vocab)

# + [markdown] colab_type="text" id="VPCdQxJ7fXDA" pycharm={"name": "#%% md\n"}
# And now let's have a look at one hot encoding on our movie reviews.

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="aQkehg82fTjv" outputId="35fb26b1-2a0c-4679-d2e7-2c76f78e58c5" pycharm={"name": "#%%\n"}
positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_encode = one_hot_encoding(positive_review)
neg_encode = one_hot_encoding(negative_review)

print("Positive:", pos_encode)
print("Negative:", neg_encode)

# + [markdown] colab_type="text" id="C5bqOjNcfjMD" pycharm={"name": "#%% md\n"}
# ### Word Embeddings
# Luckily there is a third method that is far superior, **word embeddings**. This method keeps the order of words intact as well as encodes similar words with very similar labels. It attempts to not only encode the frequency and order of words but the meaning of those words in the sentence. It encodes each word as a dense vector that represents its context in the sentence.
#
# Unlike the previous techniques word embeddings are learned by looking at many different training examples. You can add what's called an *embedding layer* to the beggining of your model and while your model trains your embedding layer will learn the correct embeddings for words. You can also use pretrained embedding layers.

# + [markdown] colab_type="text" id="FcOcD40KfsNG" pycharm={"name": "#%% md\n"}
# ### Practice with Movie Review Dataset
#
# Using Sentiment Analysis to classify movie reviews as either postive, negative or neutral.
#
# Well start by loading in the IMDB movie review dataset from keras. This dataset contains 25,000 reviews from IMDB where each one is already preprocessed and has a label as either positive or negative. Each review is encoded by integers that represents how common a word is in the entire dataset. For example, a word encoded by the integer 3 means that it is the 3rd most common word in the dataset.

# + colab={"base_uri": "https://localhost:8080/", "height": 177} colab_type="code" id="wDjZivZDfbfd" outputId="9dd55114-71d2-4d3d-9069-19512dac375e" pycharm={"name": "#%%\n"}
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="3KGTgMF_fzD1" outputId="067852b2-e7dd-428b-a230-641c581d932b" pycharm={"name": "#%%\n"}
# Lets look at one review
train_data[1]

# + [markdown] colab_type="text" id="vD_sLcyFDCLq" pycharm={"name": "#%% md\n"}
# ###  Preprocessing the Reviews
#
# We can use keras to help pad the reviews as follows:
#
# - if the review is greater than 250 words then trim off the extra words
# - if the review is less than 250 words add the necessary amount of 0's to make it equal to 250.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 507} colab_type="code" id="drzKILAjDVtN" outputId="644f0b90-a849-4a0e-9b70-f37552abf0cf" pycharm={"name": "#%%\n"}
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)
train_data[0] #Shows array with padding numbers from function

# + [markdown] colab_type="text" id="wXy2_5u4DYO_" pycharm={"name": "#%% md\n"}
# ###  Creating the Model
#
# Now it's time to create the model. We'll use a word embedding layer as the first layer in our model and add a LSTM layer afterwards that feeds into a dense node to get our predicted sentiment.
#
# 32 stands for the output dimension of the vectors generated by the embedding layer. We can change this value if we'd like!
#

# + colab={} colab_type="code" id="_cVp5DlyDW80" pycharm={"name": "#%%\n"}
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# + colab={"base_uri": "https://localhost:8080/", "height": 262} colab_type="code" id="zYnyDPkKFcgX" outputId="03ccd206-ace9-4075-e213-18a5de561c92" pycharm={"name": "#%%\n"}
model.summary()

# + [markdown] colab_type="text" id="d_UnieakGhkR" pycharm={"name": "#%% md\n"}
# ###  Training Model

# + colab={"base_uri": "https://localhost:8080/", "height": 404} colab_type="code" id="X5Pf1a6HFf1Y" outputId="827bff19-a983-468e-c6a4-b522f94fc203" pycharm={"name": "#%%\n"}
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# + [markdown] colab_type="text" id="5MIJfJOjH7Yo" pycharm={"name": "#%% md\n"}
# ###  Making Predictions

# + colab={"base_uri": "https://localhost:8080/", "height": 297} colab_type="code" id="_vGk4v8BGv1I" outputId="5cdff153-b963-41aa-e0c5-4e155c0d1522" pycharm={"name": "#%%\n"}
word_index = imdb.get_word_index()

def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)


# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="t3i-P101JTQF" outputId="18358431-7aac-4452-ce44-d3560369fa96" pycharm={"name": "#%%\n"}
# while were at it lets make a decode function to turn the integers into words

reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]
  
print(decode_integers(encoded))


# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="FCKpBDxzJ9kK" outputId="02bc3ba8-3cba-43d4-b108-996606f9e5ef" pycharm={"name": "#%%\n"}
#Making a prediction

def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  print(result[0])

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)

#The higher the number the more positive the review, lower number the more negative

# + [markdown] colab_type="text" id="TPpGJJbMMZys" pycharm={"name": "#%% md\n"}
# ## RNN Play Generator
#
# We are going to use a RNN to generate a play. We will simply show the RNN an example of something we want it to recreate and it will learn how to write a version of it on its own. We'll do this using a character predictive model that will take as input a variable length sequence and predict the next character. We can use the model many times in a row with the output from the last predicition as the input for the next call to generate a sequence

# + colab={"base_uri": "https://localhost:8080/", "height": 124} colab_type="code" id="qmk9PX3DKTLH" outputId="fc1858be-2647-46d4-8d23-336e81ce301e" pycharm={"name": "#%%\n"}
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="AzvNpJACMzxo" outputId="80ba33f6-0d36-4ded-eeff-13979bc5b40e" pycharm={"name": "#%%\n"}
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# + [markdown] colab_type="text" id="FhlySIb7NCYV" pycharm={"name": "#%% md\n"}
# ###  Loading Your Own Data
# To load your own data, you'll need to upload a file from the dialog below. Then you'll need to follow the steps from above but load in this new file instead.
#

# + colab={} colab_type="code" id="aTU0w3nBNITU" pycharm={"name": "#%%\n"}
#from google.colab import files
#path_to_file = list(files.upload().keys())[0]

# + [markdown] colab_type="text" id="AX7FMFPrNSkO" pycharm={"name": "#%% md\n"}
# ### Read Contents of File

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="sZSsjAanNPGV" outputId="78c7bab6-ab91-4cc6-c73e-6529f1938130" pycharm={"name": "#%%\n"}
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

# + colab={"base_uri": "https://localhost:8080/", "height": 279} colab_type="code" id="3x6hD8WzNlKT" outputId="c27668ef-1b6a-4543-de6a-3bf4389fda76" pycharm={"name": "#%%\n"}
# Take a look at the first 250 characters in text
print(text[:250])

# + [markdown] colab_type="text" id="11ofv9eJNz5l" pycharm={"name": "#%% md\n"}
# ###  Encoding
# Since this text isn't encoded yet well need to do that ourselves. We are going to preprocess and encode each unique character as a different integer.

# + colab={} colab_type="code" id="vWhvtucwNnEr" pycharm={"name": "#%%\n"}
vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

# + colab={"base_uri": "https://localhost:8080/", "height": 52} colab_type="code" id="FFe6CTKIN4s8" outputId="4a9b2d0c-5297-45e0-b62a-79f57d1d67fe" pycharm={"name": "#%%\n"}
# lets look at how part of our text is encoded
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))


# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="Af52YChSW5hX" outputId="3a73b596-a39c-4c2a-9384-619e81293fad" pycharm={"name": "#%%\n"}
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))

# + [markdown] colab_type="text" id="CJBkzpdFPtQi" pycharm={"name": "#%% md\n"}
# ###  Creating Training Examples
# Remember our task is to feed the model a sequence and have it return to us the next character. This means we need to split our text data from above into many shorter sequences that we can pass to the model as training examples.
#
# The training examples we will prepapre will use a seq_length sequence as input and a seq_length sequence as the output where that sequence is the original sequence shifted one letter to the right. For example:
#
#  `input: Hell` | `output: ello`
#
# Our first step will be to create a stream of characters from our text data.

# + colab={} colab_type="code" id="K8bugWcDN6b9" pycharm={"name": "#%%\n"}
seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# + [markdown] colab_type="text" id="qeNOthGSQGPE" pycharm={"name": "#%% md\n"}
# Next we can use the batch method to turn this stream of characters into batches of desired length.

# + colab={} colab_type="code" id="YGcOyd3sQF4e" pycharm={"name": "#%%\n"}
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


# + [markdown] colab_type="text" id="sZA3TKf_QRTM" pycharm={"name": "#%% md\n"}
# Now we need to use these sequences of length 101 and split them into input and output.

# + colab={} colab_type="code" id="BClPDcZHQCTt" pycharm={"name": "#%%\n"}
def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry

# + colab={"base_uri": "https://localhost:8080/", "height": 787} colab_type="code" id="OGsWlXatQZCQ" outputId="155f8453-520d-4419-f91c-9ca8ad1024ab" pycharm={"name": "#%%\n"}
#Below is an example of the above code and how it works
for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))

# + [markdown] colab_type="text" id="v6OxuFKVXpwK" pycharm={"name": "#%% md\n"}
# Finally we need to make training batches.

# + colab={} colab_type="code" id="cRsKcjhXXuoD" pycharm={"name": "#%%\n"}
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# + [markdown] colab_type="text" id="E6YRmZLtX0d0" pycharm={"name": "#%% md\n"}
# ### Building the Model
# Now it is time to build the model. We will use an embedding layer a LSTM and one dense layer that contains a node for each unique character in our training data. The dense layer will give us a probability distribution over all nodes.

# + colab={"base_uri": "https://localhost:8080/", "height": 262} colab_type="code" id="5v_P2dEic4qt" outputId="96b7c775-3f4d-4d86-bec1-6720979ecbfd" pycharm={"name": "#%%\n"}
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

# + [markdown] colab_type="text" id="8gfnHBUOvPqE" pycharm={"name": "#%% md\n"}
# ### Creating a Loss Function
# Now we are going to create our own loss function for this problem. This is because our model will output a (64, sequence_length, 65) shaped tensor that represents the probability distribution of each character at each timestep for every sequence in the batch. 
#
#

# + [markdown] colab_type="text" id="g_ERM4F15v_S" pycharm={"name": "#%% md\n"}
# However, before we do that let's have a look at a sample input and the output from our untrained model. This is so we can understand what the model is giving us.
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="KdvEqlwc6_q0" outputId="e8d0b1e2-4537-42b2-f4ab-2d88708a4fb0" pycharm={"name": "#%%\n"}
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape

# + colab={} colab_type="code" id="RQS5KXwi7_NX" pycharm={"name": "#%%\n"}
# we can see that the predicition is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)

# + colab={"base_uri": "https://localhost:8080/", "height": 279} colab_type="code" id="sA1Zhop28V9n" outputId="d040ca28-00d8-4d21-a190-5a03381736de" pycharm={"name": "#%%\n"}
# lets examine one prediction
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step

# + colab={"base_uri": "https://localhost:8080/", "height": 349} colab_type="code" id="UbIoe7Ei8q3q" outputId="404dd6c4-0972-41c9-9832-1fa3d3fdfad8" pycharm={"name": "#%%\n"}
# and finally well look at a prediction at the first timestep
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course its 65 values representing the probabillity of each character occuring next

# + colab={"base_uri": "https://localhost:8080/", "height": 54} colab_type="code" id="qlEYM1H995gR" outputId="8f05a806-edc4-402b-c5f9-b005944af40d" pycharm={"name": "#%%\n"}
# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars  # and this is what the model predicted for training sequence 1


# + [markdown] colab_type="text" id="qcCBfPjN9Cnp" pycharm={"name": "#%% md\n"}
# So now we need to create a loss function that can compare that output to the expected output and give us some numeric value representing how close the two were. 

# + colab={} colab_type="code" id="ZOw23fWq9D9O" pycharm={"name": "#%%\n"}
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# + [markdown] colab_type="text" id="kcg75GwXgW81" pycharm={"name": "#%% md\n"}
# ### Compiling the Model
# At this point we can think of our problem as a classification problem where the model predicts the probabillity of each unique letter coming next. 
#

# + colab={} colab_type="code" id="9g6o7zA_hAiS" pycharm={"name": "#%%\n"}
model.compile(optimizer='adam', loss=loss)

# + [markdown] colab_type="text" id="YgDKr4yvjLPI" pycharm={"name": "#%% md\n"}
# ### Creating Checkpoints
# Now we are going to setup and configure our model to save checkpoinst as it trains. This will allow us to load our model from a checkpoint and continue training it.

# + colab={} colab_type="code" id="v7aMushYjSpy" pycharm={"name": "#%%\n"}
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# + [markdown] colab_type="text" id="0p7acPvGja5c" pycharm={"name": "#%% md\n"}
# ### Training
# Finally, we will start training the model. 
#
# **If this is taking a while go to Runtime > Change Runtime Type and choose "GPU" under hardware accelerator.**
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="R4PAgrwMjZ4_" outputId="f383a2ee-c156-4f89-8a8b-83df0be48747" pycharm={"name": "#%%\n"}
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])

# + [markdown] colab_type="text" id="9GhoHJVtmTsz" pycharm={"name": "#%% md\n"}
# ### Loading the Model
# We'll rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one peice of text to the model and have it make a prediction.

# + colab={} colab_type="code" id="TPSto3uimSKp" pycharm={"name": "#%%\n"}
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# + [markdown] colab_type="text" id="boEJvy_vjLJQ" pycharm={"name": "#%% md\n"}
# Once the model is finished training, we can find the **lastest checkpoint** that stores the models weights using the following line.
#
#

# + colab={} colab_type="code" id="PZIEZWE4mNKl" pycharm={"name": "#%%\n"}
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# + [markdown] colab_type="text" id="CmPPtbaTKF8d" pycharm={"name": "#%% md\n"}
# We can load **any checkpoint** we want by specifying the exact file to load.

# + colab={} colab_type="code" id="YQ_5p0ehKFDn" pycharm={"name": "#%%\n"}
checkpoint_num = 50
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))


# + [markdown] colab_type="text" id="KaZWalEeAxQN" pycharm={"name": "#%% md\n"}
# ### Generating Text
# Now we can use the lovely function provided by tensorflow to generate some text using any starting string we'd like.

# + colab={} colab_type="code" id="oPSALdQXA3l3" pycharm={"name": "#%%\n"}
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 800

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
    
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      # Turn the information to back into characters
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


# + colab={"base_uri": "https://localhost:8080/", "height": 559} colab_type="code" id="cAJqhD9AA5mF" outputId="fc4b51f4-1227-4042-e80b-2ebbb863dd10" pycharm={"name": "#%%\n"}
inp = input("Type a starting string: ")
print(generate_text(model, inp))

# + [markdown] colab_type="text" id="CBjHrzzyOBVr" pycharm={"name": "#%% md\n"}
# *And* that's pretty much it for this module! I highly reccomend messing with the model we just created and seeing what you can get it to do!
