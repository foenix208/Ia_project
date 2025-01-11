
import csv

from src.algo1 import algo1
from src.algo2 import algo2
from src.algo3 import algo3

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_data(filepath):
    X, y = [], []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X.append([int(num) for num in row[:-1]])
            y.append(int(row[-1]))  # Type de main (0 à 9)
    return np.array(X), np.array(y)

def main():
    
    # Charger les données d'entraînement et de test
    X_train, y_train = load_data('data/poker-hand-training-true.data')
    X_test, y_test = load_data('data/poker-hand-testing.data')

    clf = SVC(kernel="linear")
    clf = clf.fit(X_train, y_train)

    Z = clf.predict (X_test)

    algo1()
    # algo2(X_train, X_test, y_train, y_test)
    algo3()

if __name__ == "__main__":
    main()
