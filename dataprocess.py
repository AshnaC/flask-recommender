import pandas as pd

movies = pd.read_csv("data/movies.dat", sep="::", header=None, names=['movieId', 'name', 'genre'])
ratings = pd.read_csv("data/ratings.dat", sep="::", header=None, names=['userId', 'movieId', 'rate', 'timestamp'])
users = pd.read_csv("data/users.dat", sep="::", names=['userId', 'gender', 'age', 'occupation', 'zipcode'])

movieIndx = ratings.groupby("movieId").ngroup()
ratings['movieIndx'] = movieIndx

userIndx = ratings.groupby("userId").ngroup()
ratings['userIndx'] = userIndx

movies_new = pd.merge(movies, ratings.drop_duplicates(subset=['movieId'])[['movieId', 'movieIndx']], on='movieId',
                      how="left")

full_movies = movies_new.to_json(orient='records')


def getRecommendedMovies(dist, indices):
    recc_df = pd.DataFrame(indices[0], columns=['movieIndx'])
    recc_df['dist'] = dist[0]
    movie_df =movies_new[movies_new['movieIndx'].isin(indices[0])]
    reccom_movies = pd.merge(recc_df, movie_df, how="left", on="movieIndx")
    return reccom_movies.to_json(orient='records')