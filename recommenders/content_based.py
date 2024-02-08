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
imdb = pd.read_csv('resources/data/imdb_data.csv')

# Combining tabels
df_merged = imdb_data[['movieId','title_cast','director', 'plot_keywords']]
df_merged = df_merged.merge(movies[['movieId', 'genres', 'title']], on='movieId', how='inner')

df_merged.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    #Creating a corpus
    
    
    df_merged['corpus'] = ''
    corpus = []

    # List of the columns we want to use to create our corpus 
    columns = ['title_cast', 'director', 'plot_keywords', 'genres']

    # For each movie, combine the contents of the selected columns to form its unique corpus 
    for i in range(0, df_merged.shape[0]):
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
    
    df_merged['corpus'] = df_merged['corpus'].str.replace('|', ' ')
    

# Subset of the data
    movies_subset = df_merged[:subset_size]
    return movies_subset

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
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['corpus'])
    indices = pd.DataFrame(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    
    score_series = rank_1 + rank_2 +rank_3
    # Calculating the scores
    #score_series_1 = pd.DataFrame(rank_1)
    #score_series_2 = pd.DataFrame(rank_2)
    #score_series_3 = pd.DataFrame(rank_3)
    #column = score_series_1.columns
    # Getting the indexes of the 10 most similar movies
    #series_4 = score_series_1.append(score_series_2)
    #listings = series_4.append(score_series_3).sort_values(by = column, ascending = False)
    listings = pd.Series(score_series).sort_values(ascending=False)
    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = (listings.iloc[1:50].index).to_list()
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        if (movies['title'])[i] in movie_list:
            (movies['title'])[i]
            top_n = top_n+1
        else:
            recommended_movies.append((movies['title'])[i])
    return recommended_movies
