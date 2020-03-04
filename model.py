import pandas as pd
from mf import MF

import pickle

movies = pd.read_csv("data/movies.dat", sep="::", header=None, names=['movieId', 'name', 'genre'])
ratings = pd.read_csv("data/ratings.dat", sep="::", header=None, names=['userId', 'movieId', 'rate', 'timestamp'])
users = pd.read_csv("data/users.dat", sep="::", names=['userId', 'gender', 'age', 'occupation', 'zipcode'])

movieIndx = ratings.groupby("movieId").ngroup()
ratings['movieIndx'] = movieIndx

userIndx = ratings.groupby("userId").ngroup()
ratings['userIndx'] = userIndx

movies_new = pd.merge(movies, ratings.drop_duplicates(subset=['movieId'])[['movieId', 'movieIndx']], on='movieId',
                      how="left")

ratings_mat = pd.pivot(ratings, index="userIndx", columns="movieIndx", values="rate")

model = MF(ratings_mat, alpha=0.01, reg=0.01, iterations=20, K=20)
model.train(ratings)

pickle.dump(model, open('model.pkl', 'wb'))
