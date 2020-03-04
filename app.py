from flask import Flask, request, jsonify
import pickle
from dataprocess import full_movies, getRecommendedMovies

app = Flask(__name__)

model = pickle.load(open('model.pkl', "rb"))
knn_model =pickle.load(open('knn_model.pkl', "rb"))


@app.route('/loadmovies')
def loadmovies():
    #http://127.0.0.1:5000/loadmovies
    return full_movies

@app.route('/getmovies')
def index():
    # http://127.0.0.1:5000/getmovies?movieInx=0
    movieFeatures = model.Q
    selectedMovieId = int(request.args.get('movieInx'))
    movie_feature = movieFeatures[selectedMovieId, :]
    print(movie_feature)
    dist, indices = knn_model.kneighbors([movie_feature], 20)
    movies = getRecommendedMovies(dist, indices)
    return movies

if __name__ =="__main__":
    #app.run(debug=True)
    app.run(host='127.0.0.1', port=8080, debug=True)