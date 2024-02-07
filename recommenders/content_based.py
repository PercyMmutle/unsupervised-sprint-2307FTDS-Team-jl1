"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
imdb = pd.read_csv('resources/data/imdb_data.csv')
movies.dropna(inplace=True)

# Merging the data
df_merged = imdb[['movieId','title_cast','director', 'plot_keywords']]
df_merged = df_merged.merge(movies[['movieId', 'genres', 'title']], on='movieId', how='inner')
df_merged.head()

# Standadizing the data
df_merged['title_cast'] = df_merged.title_cast.astype(str)
df_merged['plot_keywords'] = df_merged.plot_keywords.astype(str)
df_merged['genres'] = df_merged.genres.astype(str)
df_merged['director'] = df_merged.director.astype(str)

# Removing spaces between names
df_merged['director'] = df_merged['director'].apply(lambda x: "".join(x.lower() for x in x.split()))
df_merged['title_cast'] = df_merged['title_cast'].apply(lambda x: "".join(x.lower() for x in x.split()))

# Discarding the pipes between the actors' full names and getting only the first three names
df_merged['title_cast'] = df_merged['title_cast'].map(lambda x: x.split('|')[:3])
df_merged['title_cast'] = df_merged['title_cast'].apply(lambda x: ",".join(x))

# Discarding the pipes between the plot keywords' and getting only the first five words
df_merged['plot_keywords'] = df_merged['plot_keywords'].map(lambda x: x.split('|')[:5])
df_merged['plot_keywords'] = df_merged['plot_keywords'].apply(lambda x: " ".join(x))

# Discarding the pipes between the genres 
df_merged['genres'] = df_merged['genres'].map(lambda x: x.lower().split('|'))
df_merged['genres'] = df_merged['genres'].apply(lambda x: " ".join(x))



# Creating an empty column and list to store the corpus for each movie
df_merged['corpus'] = ''
corpus = []

# List of the columns we want to use to create our corpus 
columns = ['title_cast', 'director', 'plot_keywords', 'genres']

# For each movie, combine the contents of the selected columns to form its unique corpus 
for i in range(0, len(df_merged['movieId'])):
    words = ''
    for col in columns:
        # Convert to string before concatenating
        if col == 'title_cast':
            words = words + str(df_merged.iloc[i][col]).replace(",", " ") + " "
        else:
            words = words + str(df_merged.iloc[i][col]) + " "
    corpus.append(words)

# Add the corpus information for each movie to the dataframe 
df_merged['corpus'] = corpus
df_merged.set_index('movieId', inplace=True)

# Drop the columns we don't need anymore to preserve memory
df_merged.drop(columns=['title_cast', 'director', 'plot_keywords', 'genres'], inplace=True)


count = CountVectorizer()
count_matrix = count.fit_transform(df_merged['corpus'])

# Creating a similarity score matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Print cosine_sim shape
print(cosine_sim.shape)
cosine_sim


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    df1 = df_merged.reset_index()
    for movie in movie_list:
        # Extracting movie title
        title = movie
        titles = df1[title]
        indices = pd.Series(df1.index, index=df_merged['title'])
        idx = indices['title']
    
        # Similaryty score
        sim_scores = list(enumerate(cosine_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n]
        
        movie_indices = [i[0] for i in sim_scores]
 
    return titles.iloc[movie_indices]
