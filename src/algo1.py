# Fonction pour calculer l'indice de Gini
def compute_gini(y_left, y_right):
    def gini_impurity(y):
        if len(y) == 0:
            return 0
        class_counts = {}
        for label in y:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        total = len(y)
        probabilities = [count / total for count in class_counts.values()]
        return 1 - sum(p ** 2 for p in probabilities)

    left_gini = gini_impurity(y_left)
    right_gini = gini_impurity(y_right)

    total_size = len(y_left) + len(y_right)
    gini = (len(y_left) / total_size) * left_gini + \
        (len(y_right) / total_size) * right_gini

    return gini

# Fonction pour trouver la meilleure division basée sur Gini


def find_best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_value = None

    n_samples = len(X)
    n_features = len(X[0])

    for feature in range(n_features):
        possible_values = set(row[feature] for row in X)

        for value in possible_values:
            left_mask = [row[feature] <= value for row in X]
            right_mask = [not mask for mask in left_mask]

            y_left = [y[i] for i in range(n_samples) if left_mask[i]]
            y_right = [y[i] for i in range(n_samples) if right_mask[i]]

            gini = compute_gini(y_left, y_right)

            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_value = value

    return best_feature, best_value

# Construction de l'arbre de décision


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2):
    if len(set(y)) == 1:  # Si tous les labels sont identiques
        return y[0]

    if max_depth is not None and depth >= max_depth:  # Profondeur maximale atteinte
        majority_class = max(set(y), key=y.count)
        return majority_class

    if len(y) < min_samples_split:
        majority_class = max(set(y), key=y.count)
        return majority_class

    feature, value = find_best_split(X, y)
    if feature is None:  # Aucune division valide
        majority_class = max(set(y), key=y.count)
        return majority_class

    left_mask = [row[feature] <= value for row in X]
    right_mask = [not mask for mask in left_mask]

    X_left = [X[i] for i in range(len(X)) if left_mask[i]]
    y_left = [y[i] for i in range(len(y)) if left_mask[i]]
    X_right = [X[i] for i in range(len(X)) if right_mask[i]]
    y_right = [y[i] for i in range(len(y)) if right_mask[i]]

    left_tree = build_tree(X_left, y_left, depth + 1,
                           max_depth, min_samples_split)
    right_tree = build_tree(X_right, y_right, depth + 1,
                            max_depth, min_samples_split)

    return (feature, value, left_tree, right_tree)


# Prédiction avec l'arbre de décision
def predict_tree(sample, tree):
    if isinstance(tree, int):
        return tree

    feature, value, left_tree, right_tree = tree

    if sample[feature] <= value:
        return predict_tree(sample, left_tree)
    else:
        return predict_tree(sample, right_tree)

# Fonction principale pour entraîner le modèle


def algo1(X_train, X_test, y_train, y_test, max_depth=10, min_samples_split=2):
    tree = build_tree(X_train, y_train, max_depth=max_depth,
                      min_samples_split=min_samples_split)

    # Prédictions
    y_pred = [predict_tree(sample, tree) for sample in X_test]

    # Calcul de la précision
    accuracy = sum(1 for true, pred in zip(y_test, y_pred)
                   if true == pred) / len(y_test)
    print(f"Précision de l'arbre de décision : {accuracy * 100:.2f}%")
