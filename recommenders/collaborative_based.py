"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

def get_similar_movies(movie_name,similarity, user_rating):
    similar_score = similarity[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
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

    user_ids = pred_movies(movie_list)
    
    df_init_users = ratings_df.loc[ratings_df.index.isin(user_ids)]

    user_ratings = pd.merge(movies_df, df_init_users).drop(['genres'], axis=1)
    
    user_ratings = user_ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
    user_ratings = user_ratings.fillna(0)
    similarity = user_ratings.corr(method='pearson')

    recommended_movies = pd.DataFrame()
    movie_found = False
    for movie in movie_list:
        try:
            similar_movies = get_similar_movies(movie, similarity, user_rating=5)
            recommended_movies = recommended_movies.append(similar_movies.reset_index(), ignore_index=True)
            movie_found = True  # Set to True if at least one movie is found
        except KeyError as e:
            continue

    # If none of the movies are found in the similarity matrix, select top 10 movies from similarity matrix
    if not movie_found:
        top_movies = similarity.sum().nlargest(10)  # Select top 10 movies
        recommended_movies = pd.DataFrame(top_movies).reset_index()

    # Handle the case where fewer than 10 movies are found
    if len(recommended_movies) < 10:
        print("Less than 10 movies found. Recommending as many as available.")

    # Ensure recommended_movies contains exactly 10 movies
    recommended_movies = pd.Series(recommended_movies['title'])

    return recommended_movies
