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
#     language: python
#     name: python3
# ---

# + [markdown] id="uGd4NYQX1Rf_" pycharm={"name": "#%% md\n"}
# *Note: You are currently reading this using Google Colaboratory which is a cloud-hosted version of Jupyter Notebook. This is a document containing both text cells for documentation and runnable code cells. If you are unfamiliar with Jupyter Notebook, watch this 3-minute introduction before starting this challenge: https://www.youtube.com/watch?v=inN8seMm7UI*
#
# ---
#
# In this challenge, you will create a book recommendation algorithm using **K-Nearest Neighbors**.
#
# You will use the [Book-Crossings dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). This dataset contains 1.1 million ratings (scale of 1-10) of 270,000 books by 90,000 users. 
#
# After importing and cleaning the data, use `NearestNeighbors` from `sklearn.neighbors` to develop a model that shows books that are similar to a given book. The Nearest Neighbors algorithm measures distance to determine the “closeness” of instances.
#
# Create a function named `get_recommends` that takes a book title (from the dataset) as an argument and returns a list of 5 similar books with their distances from the book argument.
#
# This code:
#
# `get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")`
#
# should return:
#
# ```
# [
#   'The Queen of the Damned (Vampire Chronicles (Paperback))',
#   [
#     ['Catch 22', 0.793983519077301], 
#     ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
#     ['Interview with the Vampire', 0.7345068454742432],
#     ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
#     ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
#   ]
# ]
# ```
#
# Notice that the data returned from `get_recommends()` is a list. The first element in the list is the book title passed in to the function. The second element in the list is a list of five more lists. Each of the five lists contains a recommended book and the distance from the recommended book to the book passed in to the function.
#
# If you graph the dataset (optional), you will notice that most books are not rated frequently. To ensure statistical significance, remove from the dataset users with less than 200 ratings and books with less than 100 ratings.
#
# The first three cells import libraries you may need and the data to use. The final cell is for testing. Write all your code in between those cells.

# + id="Y1onB6kUvo4Z" pycharm={"name": "#%%\n"}
# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# + colab={"base_uri": "https://localhost:8080/"} id="iAQGqqO_vo4d" outputId="f2295207-1146-47b7-8bcc-b5defc96cf56" pycharm={"name": "#%%\n"}
# get data files
# !wget -nc https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

# !unzip -n book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# + id="NClILWOiEd6Q" pycharm={"name": "#%%\n"}
# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# + id="kaTECWVHPO6a" pycharm={"name": "#%%\n"}
df = pd.merge(df_ratings, df_books, on='isbn')

# + id="yAGcVoO3KAFC" pycharm={"name": "#%%\n"}
df_user_condition = df_ratings.groupby("user")["rating"].count().reset_index().rename(columns = {'rating':'ratingCount'})
df_user_condition = df_user_condition[df_user_condition['ratingCount'] > 200]
user_ok_list = df_user_condition.user.tolist()

# + id="XrSfxm9uuxum" pycharm={"name": "#%%\n"}
df_isbn_condition = df.groupby('isbn')['rating'].count().reset_index().rename(columns = {'rating':'ratingCount'})
df_isbn_condition = df.merge(df_isbn_condition, left_on ='isbn', right_on ='isbn', how='left')
df_isbn_condition = df_isbn_condition[df_isbn_condition['ratingCount'] > 100]

# + id="uWdP5q9APYuc" pycharm={"name": "#%%\n"}
df_ok_condition = df_isbn_condition[df_isbn_condition['user'].isin(user_ok_list)]

# + id="fxLobozrx12a" pycharm={"name": "#%%\n"}
df_pivot = df_ok_condition.pivot_table(index='title', columns='user', values='rating').fillna(0)
matrix = csr_matrix(df_pivot.values)

# + colab={"base_uri": "https://localhost:8080/"} id="XFYVyGbxG97i" outputId="9b912e0f-2126-4357-913a-efe8349eddca" pycharm={"name": "#%%\n"}
model_knn = NearestNeighbors(metric = 'cosine', n_neighbors=5, p=2, algorithm='auto')
model_knn.fit(matrix)


# + id="f5ZUd-L1SQz7" pycharm={"name": "#%%\n"}
def get_recommends(book = ""):

    for query_index in range(len(df_pivot)):
        if df_pivot.index[query_index] == book:
            break

    ret = [df_pivot.index[query_index], []]
    distances, indices = model_knn.kneighbors(df_pivot.iloc[query_index,:].values.reshape(1, -1))
    
    for i in range(1, len(distances.flatten())):
        ret[1].insert(0, [df_pivot.index[indices.flatten()[i]], distances.flatten()[i]])

    return ret


# + [markdown] id="eat9A2TKawHU" pycharm={"name": "#%% md\n"}
# Use the cell below to test your function. The `test_book_recommendation()` function will inform you if you passed the challenge or need to keep trying.

# + colab={"base_uri": "https://localhost:8080/"} id="jd2SLCh8oxMh" outputId="07dd5df2-cd33-43f6-cd79-a515a297a987" pycharm={"name": "#%%\n"}
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()
