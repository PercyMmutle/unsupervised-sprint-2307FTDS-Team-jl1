"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
movies_df = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')

st.markdown(
	"""
	<style>
	.main{
	background-color: #ADD8E6;
	color: #555555
	}
	h1, h2, h3 {
    color: #4D4D4D;
    }
	p, ol, ul, dl {
	color: #4D4D4D
	}
	</style>
	""",
	unsafe_allow_html=True
)

def header(title, subheader):
	logo_col, title_col = st.columns(2)
	image = Image.open("resources/imgs/logot.png")
	logo_col.image(image)
	title_col.title(title)
	st.markdown('<hr style="border: 2px solid #228B22;">', unsafe_allow_html=True)
	st.subheader(subheader)

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
<<<<<<< HEAD
    page_options = ["Recommender System","Solution Overview","Data Statistics","Movie Database", "Contact Us"]
=======
    page_options = ["Recommender System","Solution Overview","Data Statistics","About Us"]
>>>>>>> parent of 86f7f33 (collaborative_based filtering using pearson corr)

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        header("Movie Recommender Engine", "EXPLORE Data Science Academy Unsupervised Predict")
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
<<<<<<< HEAD
        header("Solution Overview", "Explanations of the models used")
        # Content-Based Filtering Explanation
        st.header('Content-Based Filtering:')
        st.write("""
        Content-Based Filtering recommends items based on the features or attributes of the items themselves. 
        In the context of movie recommendations, it involves analyzing the content of movies, such as genres, keywords, or other descriptive features. The basic steps involved are as follows:

        1. **Feature Extraction:**
            - Extract relevant features from the items (movies in this case). 
            - For movies, this could be genres, keywords, or any other descriptive information.

        2. **Vectorization:**
            - Convert these features into a numerical representation. 
            - Common techniques include CountVectorizer or TF-IDF (Term Frequency-Inverse Document Frequency) for natural language features like genres.

        3. **Similarity Calculation:**
            - Compute similarity scores between items based on their feature vectors. 
            - Cosine similarity is a commonly used metric for this purpose.

        4. **Recommendation:**
            - Given a user's preference or selected item, recommend items with the highest similarity scores.
        """)

        # Collaborative-Based Filtering Explanation
        st.header('Collaborative-Based Filtering:')
        st.write("""
        Collaborative-Based Filtering makes recommendations by leveraging user-item interactions and finding patterns or relationships between users and items. This method assumes that users who agreed in the past tend to agree again in the future. There are two types of collaborative-based filtering: user-based and item-based.

        1. **User-Based Collaborative Filtering:**
            - Calculate the similarity between users based on their past interactions (ratings). 
            - Similarity metrics include Cosine Similarity or Pearson Correlation.

        2. **Predict a user's rating for an item:**
            - Predict a user's rating for an item by considering the ratings of similar users.

        3. **Recommendation:**
            - Recommend items with the highest predicted ratings.

        4. **Item-Based Collaborative Filtering:**
            - Calculate the similarity between items based on user interactions.

        5. **Predict a user's rating for an item:**
            - Predict a user's rating for an item by considering the user's past ratings on similar items.

        6. **Recommendation:**
            - Recommend items with the highest predicted ratings.
        """)

        # Singular Value Decomposition (SVD) Algorithm Explanation
        st.header('Singular Value Decomposition (SVD) Algorithm:')
        st.write("""
        SVD is a matrix factorization technique commonly used in collaborative filtering. It decomposes the user-item interaction matrix into three matrices: user matrix, item matrix, and diagonal matrix. The key steps are:

        1. **Matrix Decomposition:**
            - Represent the user-item interaction matrix as the product of three matrices: U (user matrix), Σ (diagonal matrix), and V^T (transpose of the item matrix).

        2. **Dimensionality Reduction:**
            - Retain only the top-k singular values to reduce the dimensionality of the matrices.
            - This helps capture the most significant features of the user-item interactions.

        3. **Prediction:**
            - Predict user-item interactions by multiplying the reduced matrices.
            - The predicted matrix represents the estimated ratings.

        4. **Recommendation:**
            - Recommend items based on the predicted ratings.
        """)
    
    if page_selection == "Data Statistics":
        header("Data Statistics", "User Interactions and Engagement")
        st.subheader("Movie Genre Wordcloud")
        st.markdown("A captivating visualization is presented under the title \"Movie Genre Wordcloud.\" The goal is to provide users with an insightful representation of the distribution of movie genres using a Word Cloud.")
=======
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")
    
    if page_selection == "Data Statistics":
        st.title("Statistics")
        movies_df
        ratings_df

>>>>>>> parent of 86f7f33 (collaborative_based filtering using pearson corr)
        text_data = ' '.join(movies_df['genres'])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        st.image(wordcloud.to_image(), caption=f'Word Cloud for Movie Genre', use_column_width=True)

<<<<<<< HEAD
        st.subheader("User Ratings Over time")
        st.markdown("")
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

        # Extract year from the timestamp
        ratings_df['year'] = ratings_df['timestamp'].dt.year

        # Calculate average rating per year
        average_ratings_per_year = ratings_df.groupby('year')['rating'].mean()

        # Plotting the distribution of ratings over time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(average_ratings_per_year.index, average_ratings_per_year.values, marker='o', linestyle='-')
        ax.set(xlabel='Year', ylabel='Average Rating', title='Distribution of Ratings Over Time')
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)

    if page_selection == "Movie Database":
        header("Movie Database", "Movie List and ratings")
        st.title("Movie Database")
        st.subheader("Movies Data")
        movies_df
        st.subheader("Rating Data")
        ratings_df
    
    
    if page_selection == "Contact Us":
        header("Contact Us", "Get in Touch")
        # Text input for name
        name = st.text_input("Name")

        # Text input for email
        email = st.text_input("Email")

        # Text input for phone number
        phone = st.text_input("Phone")

        # Address section
        st.subheader("Address")

        # Text input for street
        street = st.text_input("Street")

        # Text input for city
        city = st.text_input("City")

        # Text input for state
        state = st.text_input("State")

        # Text input for zip code
        zip_code = st.text_input("Zip Code")

        # Text input for country
        country = st.text_input("Country")

        # Display the submitted contact details
        st.subheader("Submitted Contact Details:")
        st.write(f"Name: {name}")
        st.write(f"Email: {email}")
        st.write(f"Phone: {phone}")
        st.write("Address:")
        st.write(f"Street: {street}")
        st.write(f"City: {city}")
        st.write(f"State: {state}")
        st.write(f"Zip Code: {zip_code}")
        st.write(f"Country: {country}")
=======
        # Distribution of ratings
        plt.figure(figsize=(12, 7))
        ax = sns.countplot(x='rating', data=train_df)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.title('Distribution of Ratings in train.csv')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()
    
    if page_selection == "About Us":
        st.title("Meet the Team")
>>>>>>> parent of 86f7f33 (collaborative_based filtering using pearson corr)

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
