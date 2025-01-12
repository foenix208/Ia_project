
import random
import numpy as np
import copy
import pickle

import datetime

VERSION = 1.2


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iter=1000):
        self.version = VERSION
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iter
        self.w = None
        self.b = None

        self.X = None
        self.y = None

    def complete(self):
        try:
            if self.version != VERSION:
                return False
        except AttributeError:
            return False

        return True

    def fit(self, X, y, folder, name):

        # check if model was already trained
        
        model_file_name = f"{name}"
        model_filepath = f"{folder}{model_file_name}"

        try:
            data = SVM.load(model_filepath)

            self.X = data.X
            self.y = data.y
            self.version = data.version
            self.w = data.w
            self.b = data.b

            print(f"Model loaded from file: {model_filepath}.")
            if not self.complete():
                self.version = VERSION
                print(f"Model incomplete.")
            else: return

        except FileNotFoundError:
            print(f"Model file not found. Begining training of model : {model_file_name}.")

        # fit

        t_a = datetime.datetime.now()
                    
        n_samples, n_features = X.shape

        self.X = X
        self.y = y

        self.w = np.zeros(n_features)
        self.b = 0

        print("Begining training...")

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * \
                        (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * \
                        (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

            print(f"{((i + 1) /self.n_iters * 100):.2f}% ...")

        print ("Training complete.")
        print (f"{ (datetime.datetime.now() - t_a).seconds } s elapsed")

        self.save(model_filepath)
        print(f"Model saved under '{model_filepath}'.")

    def predict(self, X_t):
        approx = np.dot(X_t, self.w)
        return np.sign(approx)
    
    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    def load(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
        
        
def fit_multi_class(X, y, folder, prefix):
    k = len(np.unique(y))

    models = []

    for i in range(k):
        Xs, ys = X, copy.copy(y)

        ys[y != i] = -1
        ys[y == i] = +1

        model = SVM()

        model_file_name = f"{prefix}_model_feature_{i}"

        model.fit(Xs, ys, folder, model_file_name)

        models.append(model)

    return models


def predict_multi_class(X, Clfs):
    N = X.shape[0]

    # len(Clfs) is supposed to be self.k if it was in an SVM class, so the number of class
    preds = np.zeros((N, len(Clfs)))

    for i, clf in enumerate(Clfs):
        pred = clf.predict(X)

        # print (f"[{i}] max: {pred.max(axis=0)}, min: {pred.min(axis=0)}")

        preds[:, i] = pred

    # get the argmax and the corresponding score
    return np.argmax(preds, axis=1), np.max(preds, axis=1), preds


def binary_convert(x):
    return [int(x) for x in list(f"{x:04b}")]


def encode(X):
    return np.array([np.array([binary_convert(x) for x in row]).flatten() for row in X])


def algo2(X_train, X_test, y_train, y_test):
    # X_train_encode = encode(X_train)
    # X_train_encode[X_train_encode == 0] = -1
    # X_test_encode = encode(X_test)
    # X_test_encode[X_test_encode == 0] = -1

    models = fit_multi_class(
        X_train, y_train, "data/models/svm/", "poker_hands")

    # Prédire sur les données de test
    # y_pred = svm.predict(X_test)
    y_pred = predict_multi_class(X_test, models)

    # Calculer la précision
    accuracy = sum(1 for i in range(len(y_test))
                   if y_test[i] == y_pred[0][i]) / len(y_test)
    print(f'Accuracy: {accuracy:.2f}')
