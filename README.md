# Ia_project


#venv
python3 -m venv .env 
source .env/bin/activate


# Classification des Mains de Poker 🃏

Ce projet implémente un programme Python qui utilise trois algorithmes de classification différents pour prédire la catégorie d'une main de poker. Les algorithmes utilisés sont :

- **Arbre de Décision**
- **Support Vector Machine (SVM)**
- **Régression Logistique**

Le programme s'appuie sur un jeu de données décrivant des mains de poker et leurs catégories.

---

## Table des Matières

1. [Description](#description)
2. [Installation](#installation)
3. [Utilisation](#utilisation)
4. [Structure du Projet](#structure-du-projet)
5. [Algorithmes Utilisés](#algorithmes-utilisés)
6. [Licence](#licence)

---

## Description

Ce projet vise à résoudre un problème de classification supervisée à l'aide d'algorithmes de machine learning. Le jeu de données utilisé contient des informations sur des mains de poker, telles que les cartes et leur valeur, et la catégorie correspondante de la main (ex. : Paire, Quinte Flush, etc.). 

L'objectif est d'entraîner des modèles pour prédire avec précision la catégorie d'une main de poker donnée.

---

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/poker-classification.git
   cd poker-classification

Utilisation

    Placez le jeu de données au format .csv dans le répertoire data. Le fichier doit contenir les colonnes décrivant les cartes et les étiquettes correspondantes.

    Lancez le programme principal pour entraîner les modèles et effectuer des prédictions :

python main.py

Les résultats des modèles (précision, matrice de confusion, etc.) seront affichés dans la console et sauvegardés dans le répertoire results.

Structure du Projet

Voici un aperçu de l'organisation des fichiers :
```
poker-classification/
│
├── data/
│   └── poker_dataset.csv     # Jeu de données (à ajouter)
│
├── models/
│   ├── decision_tree.py      # Implémentation de l'Arbre de Décision
│   ├── svm.py                # Implémentation de SVM
│   └── logistic_regression.py # Implémentation de la Régression Logistique
│
├── results/
│   └── model_results.txt     # Résultats des modèles
│
├── main.py                   # Script principal
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation du projet

```

Algorithmes Utilisés
1. Arbre de Décision

Un algorithme basé sur une structure arborescente pour prendre des décisions en fonction des caractéristiques des données. Il est simple et interprétable.
2. Support Vector Machine (SVM)

Un algorithme de classification puissant qui trouve une frontière optimale entre les classes dans un espace à haute dimension.
3. Régression Logistique

Un modèle statistique utilisé pour modéliser la probabilité qu'une observation appartienne à une classe particulière.