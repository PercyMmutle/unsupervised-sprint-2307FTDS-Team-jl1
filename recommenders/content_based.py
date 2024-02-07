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
# Import Packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Import Data
movies_data = pd.read_csv('movies.csv') # Loading movies dataset
imdb_data = pd.read_csv('imdb_data.csv') # Loading imdb dataset


# Combining tables
df_merged = imdb_data[['movieId','title_cast','director', 'plot_keywords']]
df_merged = df_merged.merge(movies_data[['movieId', 'genres', 'title']], on='movieId', how='inner')

# Creating a copus
# Creating an empty column and list to store the corpus for each movie
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

# Create a TfidfVectorizer and Remove stopwords
tfidf = TfidfVectorizer(stop_words='english')
# Fit and transform the data to a tfidf matrix
tfidf_matrix = tfidf.fit_transform(df_merged['corpus'])



# Compute the cosine similarity between each movie description
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


#-----------------------------------------------------------------------------------------------------------------------
# Creating a function 
indices = pd.Series(df_merged.index, index=df_merged['title']).drop_duplicates()


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
    title = movie_list[0]
    idx = indices[title]
# Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
# Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# Get the scores of the 10 most similar movies
    top_similar = sim_scores[1:num_recommend+1]
# Get the movie indices
    movie_indices = [i[0] for i in top_similar]
# Return the top 10 most similar movies
    return df_merged['title'].iloc[movie_indices]
