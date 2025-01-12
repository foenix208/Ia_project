import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

#* Creation de la class Logistic_regression
class Logistic_regression():
    def __init__(self, x, seed=123):
        m, n = x.shape
        np.random.seed(seed)
        self._w = np.random.rand(n + 1, 1)  # Poids (initialisation aléatoire)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Fonction sigmoïde
    
    def predict(self, x):
        m, n = x.shape
        x_1 = np.hstack((np.ones((m, 1)), x))  
        mul = np.dot(x_1, self._w)
        return self.sigmoid(mul)  # Retourner les prédictions sigmoïdes

    def compute_cost(self, y, y_hat):
        m, _ = y.shape
        # Calcul du coût avec la somme des erreurs quadratiques
        return - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))  

    def fits(self, x, y, learning_rate=0.001, num_iters=2000):
        m, n = x.shape
        x_1 = np.hstack((np.ones((m, 1)), x))  
        J_history = np.zeros(num_iters)

        for i in range(num_iters):
            # Calcul des prédictions
            predictions = self.predict(x)
            
            # Mise à jour des poids (descente de gradient)
            self._w = self._w - (learning_rate / m) * np.dot(x_1.T, (predictions - y))
            
            # Calcul du coût à chaque itération et mise à jour de J_history
            J_history[i] = self.compute_cost(y, predictions)

        return J_history  # Historique du coût , 

def algo3():

    data = pd.read_csv('data/poker-hand-training-true.data')
    prediction = [0]
    #i = 0

    #! Boucle FOR 
    for i in range(8,9):
        cp = data.copy()  # Create a copy of the data
        cp.iloc[:, -1] = cp.iloc[:, -1].apply(lambda x: 1 if x != i else 0)  # Modify the last column
        
        #*Seppart les données 
        x = cp.iloc[:, :-1]
        y = cp.iloc[:, -1:]

        #* Normalisation des donnée 
        x_mean = means = np.mean(x, axis=0) 
        x_std = np.std(x, axis=0)
        x_norm = (x - x_mean) / x_std

        #* entrainement de l'algo 
        log = Logistic_regression(x_norm)
        history = log.fits(x_norm, y, learning_rate=0.01, num_iters=2500)

        # Prédictions sur les données normalisées
        predictions = log.predict(x_norm)

        # Convertir les prédictions en valeurs binaires (0 ou 1)
        predictions_binary = (predictions >= 0.5).astype(int)

        if len(predictions_binary) == 0 :
            prediction = predictions_binary
        else : 
            for t in range(len(predictions_binary)):
                if predictions_binary[t] == 0:
                    prediction[t] = i


    #print(prediction)