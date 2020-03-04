import numpy as np


class MF:

    def __init__(self, R, alpha=0.1, reg=0.01, iterations=20, K=20):
        self.R = R
        self.alpha = alpha
        self.reg = reg
        self.iterations = iterations
        self.K = K
        self.userCount, self.movieCount = R.shape
        self.P = np.random.normal(scale=1./self.K, size=(self.userCount, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.movieCount, self.K))
        self.meanRate = np.nanmean(R, axis=0)
        self.userBias = np.zeros(self.userCount)
        self.itemBias = np.zeros(self.movieCount)

    def train(self, rating_df):
        for i in range(self.iterations):
            print('iteration start', i)
            self.sgd(rating_df)
            print('iteration end',i)

    def sgd(self, rating_df):
        for index, row in rating_df.iterrows():
            user = row['userIndx']
            movie = row['movieIndx']
            rate = row['rate']
            prediction = self.P[user, :].dot(self.Q[movie, :]) + self.meanRate[movie] + self.userBias[user] + self.itemBias[movie]
            # Find error
            err = rate - prediction
            self.P[user, :] = self.P[user, :] + 2 * self.alpha * (err * self.Q[movie, :] - self.reg * self.P[user, :])
            self.Q[movie, :] = self.Q[movie, :] + 2 * self.alpha * (err * self.P[user, :] - self.reg * self.Q[movie, :])
            self.userBias[user] = self.userBias[user] + 2 * self.alpha * (err - (self.reg * self.userBias[user]))
            self.itemBias[movie] = self.itemBias[movie] + 2 * self.alpha * (err - (self.reg * self.itemBias[movie]))

    def fullPrediction(self):
        full_rating = self.P.dot(self.Q.T) + self.meanRate + self.userBias[:, np.newaxis] + self.itemBias[np.newaxis, :]
        return full_rating
