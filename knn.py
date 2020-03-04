from sklearn.neighbors import NearestNeighbors
import pickle

model = pickle.load(open('model.pkl', "rb"))

knn = NearestNeighbors(metric="cosine", algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(model.Q)

pickle.dump(knn, open('knn_model.pkl', 'wb'))