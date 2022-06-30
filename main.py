# -*- coding: utf-8 -*-

"""

Created on Tue Nov 17 21:40:41 2020



@author: win10

"""



# 1. Library imports

import uvicorn

from fastapi import FastAPI



import numpy as np

import pickle

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# 2. Create the app object

app = FastAPI()

pickle_in = open("classifier2.pkl","rb")
count_matrix=pickle.load(pickle_in)

pickle_in = open("df.pkl","rb")
df=pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000

@app.get('/')

def index():

    return {'message': 'Hello, World'}



# 4. Route with a single parameter, returns the parameter within a message

#    Located at: http://127.0.0.1:8000/AnyNameHere

@app.get('/{name}')

def get_name(name: str):

    return {'hey'+'{name}'+ 'welcome to movie recommender'}



# 3. Expose the prediction functionality, make a prediction from the passed

#    JSON data and return the predicted Bank Note with the confidence

@app.get('/Recommend movies/{data}')

def predict_banknote(data:str):

            

            query = data

            show =[]
            


           

            cosine_sim = cosine_similarity(count_matrix)



            def get_title_from_index(index):

                return df[df.index == index]["title"].values[0]



            def get_index_from_title(title):

                return df[df.title == title]["index"].values[0]



            movie_user_likes = query

            movie_index = get_index_from_title(movie_user_likes)

            similar_movies = list(enumerate(cosine_sim[

                                                movie_index]))  # accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it

            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

            sort_size = len(sorted_similar_movies)

            if sort_size > 10:

                print("Top 10 similar movies to " + movie_user_likes + " are:\n")

                for element in sorted_similar_movies[:2]:

                    shows = (get_title_from_index(element[0]))

                    show.append(shows)

            else:

                print("Top " + str(sort_size) + "similar movies to " + movie_user_likes + " are:\n")

                for element in sorted_similar_movies[:sort_size]:

                    shows = (get_title_from_index(element[0]))

                    show.append(shows)


            return show



# 5. Run the API with uvicorn

#    Will run on http://127.0.0.1:8000

#if __name__ == '__main__':

 #   uvicorn.run(app, host='127.0.0.1', port=8000)

    

#uvicorn app:app --reload
