import numpy as np
import pandas as pd


class LinearRegressor(object):

    def __init__(self, features, prediction, learning_rate, regularize=0.0, weight_range=1, validation_set=None):
        """
        :param features: a list of columns which we want to train on
        :param prediction: a string representing the column of a DataFrame which we want to predict
        :param learning_rate: a float
        :param regularize: A scalar for the regularization term
        """

        self.features = features
        self.prediction = prediction
        self.learning_rate = learning_rate
        self.regularize = regularize
        self.sseData = None
        self.reportingSse = None
        self.validationSse = None

        self.validation_data = validation_set

        # Weights are randomly initialized between -1 and 1
        self.w = pd.Series(np.random.rand(len(self.features)) * 2 - 1, index=self.features) * weight_range


    def compute_loss(self, data, w=None, for_reporting=False):
        if w is None:
            w = self.w

        y = data[self.prediction]
        x = data[self.features]

        pred = x.dot(w)
        if not for_reporting:
            return 0.5 * (((y - pred)**2).mean() + self.regularize * np.linalg.norm(w)**2)
        else:
            return ((y - pred)**2).mean()


    def compute_average_gradient(self, data, w=None):
        if w is None:
            w = self.w

        y = data[self.prediction]
        x = data[self.features]

        pred = x.dot(w)
        error = pred - y

        # Logic to disable regularization of the bias coefficient
        w_mod = w.copy()
        if 'dummy' in w_mod.index:
            w_mod['dummy'] = 0

        avg_gradient = (error * x.T).mean(axis=1) + self.regularize * w_mod
        return avg_gradient

    def train(self, data, stop_threshold, max_iter=None):
        if max_iter is not None:
            self.sseData = np.zeros(max_iter)
            self.reportingSse = pd.Series(index=np.arange(0, max_iter))
            if self.validation_data is not None:
                self.validationSse = pd.Series(index=np.arange(0, max_iter))

        i = 0
        while True:
            gradient = self.compute_average_gradient(data)
            self.w -= self.learning_rate * gradient

            if max_iter is not None:
                self.sseData[i] = self.compute_loss(data)
                self.reportingSse[i] = self.compute_loss(data, for_reporting=True)
                if self.validation_data is not None:
                    self.validationSse[i] = self.compute_loss(self.validation_data, for_reporting=True)

            i += 1

            if np.log10(np.abs(gradient)).mean() > 80.0:
                print('Gradient is blowing up! Terminating... (Iteration {})'.format(i))
                break

            norm = np.linalg.norm(gradient)
            #print(norm)

            if norm < stop_threshold:
                print('Convergence threshold met after {} iterations!'.format(i))
                break

            if max_iter is not None and i >= max_iter:
                print('Hit max iterations!')
                break
        if max_iter is not None:
            self.sseData = self.sseData[:i]


    def predict(self, data):
        x = data[self.features]
        return x.dot(self.w)


