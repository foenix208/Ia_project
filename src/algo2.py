
import random
import numpy as np
import copy
import pickle

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array (X)
        y = np.array (y)

        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        print ("Begining training...")

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

            print (f"{((i + 1) /self.n_iters * 100):.2f}% ...")

        print ("Training complete.")

    def predict(self, X_t):
        x_s, y_s = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        s, y, X = self.s[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]

        b = y_s - np.sum(s * y * self.kernel(X, x_s, self.k), axis = 0)
        score = np.sum(s * y * self.kernel(X, X_t, self.k), axis = 0) + b

        return np.sign(score).astype(int), score
    
    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    def load(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)

def fit_multi_class(X, y, folder, prefix):
    k = len(np.unique(y))

    X = np.array (X)
    y = np.array (y)

    models = []

    for i in range (k):
        Xs, ys = X, copy.copy(y)

        ys[ys != i] = -1
        ys[ys == i] = +1

        model = SVM()

        model_file_name = f"{prefix}_model_feature_{i}"
        model_filepath = f"{folder}{model_file_name}"

        try:
            model = SVM.load(model_filepath)
            print(f"Model loaded from file: {model_filepath}.")
        except FileNotFoundError:
            print(f"Model file not found. Begining training of model : {model_file_name}.")
            model.fit(Xs, ys)
            model.save(model_filepath)
            print(f"Model saved under '{model_filepath}'.")

        models.append(model)

    return models

def predict_multi_class(X, Clfs):
    X = np.array(X)
    N = X.shape[0]

    # len(Clfs) is supposed to be self.k if it was in an SVM class, so the number of class
    preds = np.zeros((N, len(Clfs)))

    for i, clf in enumerate(Clfs):
        _, preds[:, i] = clf.predict(X)
    
    # get the argmax and the corresponding score
    return np.argmax(preds, axis=1), np.max(preds, axis=1)


def algo2(X_train, X_test, y_train, y_test):
    # Initialiser et entraîner le modèle SVM
    # svm = SVM()
    # svm.fit(X_train, y_train)

    models = fit_multi_class (X_train, y_train, "data/models/svm/", "poker_hands")

    # Prédire sur les données de test
    #y_pred = svm.predict(X_test)
    y_pred = predict_multi_class (X_test, models)

    # Calculer la précision
    accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / len(y_test)
    print(f'Accuracy: {accuracy:.2f}')
