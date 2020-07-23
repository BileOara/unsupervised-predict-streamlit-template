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

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    
    # Creating sidebar
    # you can create multiple pages this way
    st.sidebar.title("Menu")
    page_selection = st.sidebar.radio(label="", options=["Recommender System","Information", "EDA and Insights", "Prediction", "Conclusion"])


    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    
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
    if page_selection == "Information":
        st.title("Information")

        st.image('resources/imgs/Information.png',use_column_width=True)
        st.title("Project Overview")
        
        st.markdown("Machine Learning (ML) is a subset of Artificial Intelligence (AI), dating as far back as 1959, where computers are trained to pick up patterns and make decisions with little to no human interference. There are two main types, supervised and supervised learning. Supervised ML algorithms are far more flexible as the datasets used do not provide label values, the computer tries to make sense of the data to compute results. They can be used to build recommender systems.")
        st.markdown("A recommender system is an engine or platform that can predict certain choices based on responses provided by users. A great example would be a system on a streaming platform that recommends a movie or show for a user to watch found on their previous viewings or the viewings of other users that have watching habits similar to them. With the increasing use of web services such as Netflix, Showmax, YouTube amongst a few, there is an unfathomable amount of content. It would be a tedious task for a user to search through it all for things that they would enjoy watching. They are also used in other services such as online shoppping stores and networking spaces like LinkedIn.")
        st.markdown("A recommender system enhances a user's experience as the luxury of recommendations will save the user the time and effort of having to search through a large catalogue for movies that they would enjoy. This allows for the user to also be exposed to new content, creating an opportunity for further streaming because they are giving an option of content that is meaningful and desireable to them. In fact, most companies make a bulk of their revenue from recommendations. The rating functionality also assists in collecting data that can help the streaming platform establish trends and gather insights from what their users are consuming. This can assist in better content selection and marketing.")
        st.title("Problem Statement")
        st.markdown("Build a recommendation algorithm that will use a user's historical preferences to accuartely predict the rating that they will give a movie that they haven't watched.")

        


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "EDA and Insights":



        
        my_dataset = 'resources/data/train.csv'
   

        st.title("Exploratory Data Analysis")

        st.image('resources/imgs/EDA.png',use_column_width=True)

        st.write("Short explanation of what eda is")
        st.markdown("Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and diagramatic representations")
        
           # Show Dataset
        if st.checkbox("Preview DataFrame"):
            if st.button("Head"):
                 st.write(my_dataset.head())
            if st.button("Tail"):
                 st.write(my_dataset.tail())
            else:
                 st.write(my_dataset.head(2))


            # Show Entire Dataframe
        if st.checkbox("Show All DataFrame"):
         st.dataframe(my_dataset)
            # Show All Column Names
        if st.checkbox("Show All Column Name"):
         st.text("Columns:")
         st.write(my_dataset.columns)
            # Show Dimensions and Shape of Dataset
        data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
        if data_dim == 'Rows':
         st.text("Showing Length of Rows")
         st.write(len(my_dataset))
        if data_dim == 'Columns':
         st.text("Showing Length of Columns")
         st.write(my_dataset.shape[1])
          # Show Summary of Dataset
        if st.checkbox("Show Summary of Dataset"):
         st.write(my_dataset.describe())


        st.title("Visualisations")


         # Number of ratings per movie
         # ratings_per_movie
         # ratings_per_user
         # yearly_rating





            


         #my_dataset = 'resources/train.csv'


    if page_selection == "Prediction":
        st.title("Actual Model ")
        st.image('resources/imgs/actua.png',use_column_width=True)
        st.write("Details behind how the recommender system page works and some code illustrations")
        st.write("explain what is and why using both content and collarative filltering was the best solution")
        st.write("also the algorithms e.g SVD,KNN etc")
        st.write("this page has to be more techinical than page 1")

    if page_selection == "Conclusion":
        st.title("Conclusion")
        st.image('resources/imgs/conclusion.png',use_column_width=True)
        st.write("Mention why we think our solution is unique from the rest of the recommendation systems out there")
        st.write("basically what makes us stand out")
        ##ml_img = Image.open("resources/imgs/ml_img.png")
            #st.image(ml_img, use_column_width=True)
 
            #st.info("Here you will find a little more technical info on the models available for prediction")

            #tech_inf = markdown(open('resources/vector_model_exp.md').read())
            #st.markdown(tech_inf, unsafe_allow_html=True)





if __name__ == '__main__':
    main()
