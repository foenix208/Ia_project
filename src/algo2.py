
import random
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iter=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.w = None
        self.b = None
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.classes = list(set(y))  # Classes uniques
        n_classes = len(self.classes)

        # Initialiser les poids et biais pour chaque classe
        self.w = [[0] * n_features for _ in range(n_classes)]
        self.b = [0] * n_classes

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                for class_idx, class_label in enumerate(self.classes):
                    y_label = 1 if y[idx] == class_label else -1
                    condition = y_label * (sum(self.w[class_idx][j] * x_i[j] for j in range(n_features)) + self.b[class_idx]) >= 1
                    if condition:
                        for j in range(n_features):
                            self.w[class_idx][j] -= self.learning_rate * (2 * self.lambda_param * self.w[class_idx][j])
                    else:
                        for j in range(n_features):
                            self.w[class_idx][j] -= self.learning_rate * (2 * self.lambda_param * self.w[class_idx][j] - x_i[j] * y_label)
                        self.b[class_idx] -= self.learning_rate * y_label


    def fit_fast(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialiser les poids et biais pour chaque classe
        self.w = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        for i in range(self.n_iter):
            for idx, x_i in enumerate(X):
                for class_idx, class_label in enumerate(self.classes):
                    y_label = 1 if y[idx] == class_label else -1
                    condition = y_label * (np.dot(x_i, self.w[class_idx]) + self.b[class_idx]) >= 1
                    if condition:
                        self.w[class_idx] -= self.learning_rate * (2 * self.lambda_param * self.w[class_idx])
                    else:
                        self.w[class_idx] -= self.learning_rate * (2 * self.lambda_param * self.w[class_idx] - y_label * x_i)
                        self.b[class_idx] -= self.learning_rate * y_label

            print (f"Stade : {i} / {self.n_iter} ==> {(i/self.n_iter * 100):.2f}%")


    def fit_faster(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialiser les poids et biais pour chaque classe
        self.w = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        print ("Begining training...")

        for i in range(self.n_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Calcul des marges et des mises à jour
                for class_idx, class_label in enumerate(self.classes):
                    y_label_batch = np.where(y_batch == class_label, 1, -1)
                    margins = y_label_batch * (np.dot(X_batch, self.w[class_idx]) + self.b[class_idx])
                    condition = margins < 1

                    dw = (self.lambda_param * self.w[class_idx]) - np.dot(y_label_batch[condition], X_batch[condition]) / self.batch_size
                    db = -np.sum(y_label_batch[condition]) / self.batch_size

                    # Mise à jour des poids et biais pour cette classe
                    self.w[class_idx] -= self.learning_rate * dw
                    self.b[class_idx] -= self.learning_rate * db

            print (f"{((i + 1) /self.n_iter * 100):.2f}% ...")

        print ("Training complete.")

    def predict(self, X):
        predictions = []
        for x_i in X:
            scores = [sum(self.w[class_idx][j] * x_i[j] for j in range(len(x_i))) + self.b[class_idx] for class_idx in range(len(self.classes))]
            predictions.append(self.classes[scores.index(max(scores))])
        return predictions
    
    def predict_numpy(self, X):
        X = np.array(X)
        # Calcul des scores pour chaque classe
        scores = np.dot(X, self.w.T) + self.b
        # Pour chaque échantillon, on sélectionne la classe avec le score maximal
        predictions = np.argmax(scores, axis=1)
        # Convertir l'index de la classe prédite en label de classe
        return [self.classes[pred] for pred in predictions]


def algo2(X_train, X_test, y_train, y_test):
    # Initialiser et entraîner le modèle SVM
    svm = SVM()
    svm.fit_faster(X_train, y_train)

    # Prédire sur les données de test
    y_pred = svm.predict_numpy(X_test)

    # Calculer la précision
    accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / len(y_test)
    print(f'Accuracy: {accuracy:.2f}')
