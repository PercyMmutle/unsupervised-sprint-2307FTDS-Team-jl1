# Streamlit dependencies
import streamlit as st

# Set page background color
st.set_page_config(layout="wide", page_title="Movie Recommender Engine", page_icon=":movie_camera:", initial_sidebar_state="expanded", background_color="#191970")  # Midnight blue color

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # Customizing sidebar with CSS
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #191970;
            color: white;
        }
        .sidebar .sidebar-content .block-container {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","The Magic of Streamcon Picks", "About Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
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
                    st.error("Oops! Looks like this algorithm doesn't work.\
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
                    st.error("Oops! Looks like this algorithm doesn't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "The Magic of Streamcon Picks":
        st.title("The Magic of Streamcon Picks")
        st.write("With Streamcon, we believe the journey matters just as much as the destination. That's why we go beyond simple recommendations, leveraging cutting-edge technology and your unique preferences to craft a bespoke viewing experience. No more algorithms throwing darts in the dark - we unlock the magic of movies that resonate with your soul.")
        st.image('resources/images/streamcon_image.jpg', use_column_width=True)

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "About Us":
        st.title("About Us")
        st.write("Meet Our Team:")
        st.image('resources/images/percy_mmutle.png', caption='Percy Mmutle - Solutions Architect', use_column_width=True)
        st.image('resources/images/sinothabo_zwane.png', caption='Sinothabo Zwane - DevOps Engineer', use_column_width=True)
        st.image('resources/images/dakalo_mudimeli.png', caption='Dakalo Mudimeli - Data Engineer', use_column_width=True)
        st.image('resources/images/ntombenhle_nkosi.png', caption='Ntombenhle Nkosi - Data Scientist', use_column_width=True)
        st.image('resources/images/katlego_mbewe.png', caption='Katlego Mbewe - Data Analyst', use_column_width=True)
        st.write("Contact Us")
        st.write("---")
        st.write("Find our code on GitHub:")
        st.write("[Unsupervised Learning Sprint - Team JL1 App](https://github.com/PercyMmutle/unsupervised-sprint-2307FTDS-Team-jl1-App) ![GitHub](https://img.shields.io/github/stars/PercyMmutle/unsupervised-sprint-2307FTDS-Team-jl1-App?style=social)")


    # Footer
    st.write("---")
    st.write("Don't just watch it, live it! We'll find your next cinematic adventure ❤️ Cloudnet Solutions 2024")


if __name__ == '__main__':
    main()
