
import numpy as np
import faiss
import pandas as pd
from sklearn.metrics import accuracy_score
from helpers import plot_decision_boundaries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()




class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        #### YOUR CODE GOES HERE ####
        #each training instance (xi, yi) do Calculate the distance di between xi and xtest.
        table, indices = self.knn_distance(X)

        #for each class c do Count the occurrences of c in the selected k instances.
        nearest_labels = self.Y_train[indices]

        #return Assign the class label yˆtest to the one with the highest count
        predictions = []
        for labels in nearest_labels:
            predictions.append(np.bincount(labels).argmax())
        return np.array(predictions)

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2
        for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
	    #### YOUR CODE GOES HERE ####
        table, indices = self.index.search(X, self.k)
        return table, indices




def knn_5():

    X = pd.read_csv("train.csv")
    Y = pd.read_csv("test.csv")

    X_train = X.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    Y_train = X.iloc[:, -1].values   # Dernière colonne (les étiquettes)

    X_test = Y.iloc[:, :-1].values
    Y_test = Y.iloc[:, -1].values

    results = []

    # Tester toutes les combinaisons de k et de métriques de distance
    for distance_metric in ['l1', 'l2']:
        for k in [1, 10, 100, 1000, 3000]:
            #print(f"Training kNN with k={k}, distance_metric={distance_metric}")

            knn = KNNClassifier(k=k, distance_metric=distance_metric)
            knn.fit(X_train, Y_train)

            Y_pred = knn.predict(X_test)

            accuracy = accuracy_score(Y_test, Y_pred)
            #print(f"Accuracy: {accuracy}")

            results.append((k, distance_metric, accuracy))


    results_df = pd.DataFrame(results, columns=['k', 'distance_metric', 'accuracy'])


    pivot_table = results_df.pivot(index='distance_metric', columns='k', values='accuracy')

    print("Table of results :")
    print(pivot_table)

    l2_results = [res for res in results if res[1] == 'l2']
    kmax = max(l2_results, key=lambda x: x[2])[0]
    kmin = min(l2_results, key=lambda x: x[2])[0]

    model1 = KNNClassifier(k=kmax, distance_metric='l2')
    model1.fit(X_train, Y_train)
    plot_decision_boundaries(model1, X_test, Y_test, title="L2, k = kmax")

    # Modèle 2 : L2, k = kmin
    model2 = KNNClassifier(k=3000, distance_metric='l2')
    model2.fit(X_train, Y_train)
    plot_decision_boundaries(model2, X_test, Y_test, title="L2, k = kmin")

    # Modèle 3 : L1, k = kmax
    model3 = KNNClassifier(k=kmax, distance_metric='l1')
    model3.fit(X_train, Y_train)
    plot_decision_boundaries(model3, X_test, Y_test, title="L1, k = kmax")


def knn_anomaly():
    train_data = pd.read_csv('train.csv').values
    test_data= pd.read_csv('AD_test.csv').values

    # Extraire les caractéristiques
    train_features = train_data[:, :-1]
    test_features = test_data[:]

    knn = KNNClassifier(k=5, distance_metric='l2')
    knn.fit(train_features, None)
    table, indices = knn.knn_distance(test_data)
    anomaly_scores = np.sum(table, axis=1)
    top_anomalies_indices = np.argsort(anomaly_scores)[-50:]
    anomalies = np.zeros(test_features.shape[0], dtype=bool)
    anomalies[top_anomalies_indices] = True


    # shapefile_path = ('/Users/itzhakfagebaume/Desktop/IML/ne_110m_admin_1_states_provinces/n'
    #                   'e_110m_admin_1_states_provinces.shp')
    # world = gpd.read_file(shapefile_path)
    # # Extraire les États-Unis
    # us = world

    # Créer un graphique
    plt.figure(figsize=(10, 6))

    # # Afficher la carte des États-Unis (en arrière-plan)
    # us.plot(ax=plt.gca(), color='lightgray', edgecolor='black', alpha=1)

    # Afficher les points d'entraînement en noir (alpha=0.01 pour une faible opacité)
    plt.scatter(train_features[:, 0], train_features[:, 1], color='black', alpha=0.01, label="Train Data")

    # Afficher les points de test en bleu pour les points normaux et en rouge pour les anomalies
    plt.scatter(test_features[~anomalies, 0], test_features[~anomalies, 1], color='blue', label="Normal Points")
    plt.scatter(test_features[anomalies, 0], test_features[anomalies, 1], color='red', label="Anomalous Points")

    # Ajouter des labels et une légende
    plt.title("Test Data with Anomalies Highlighted")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # Afficher le graphique
    plt.show()

#########################################
def plot_decision(X, y, stumps, n_stumps, num_classes):
    # Prédictions à partir des stumps jusqu'à l'itération n_stumps
    predictions = predict_with_stumps(stumps, X, num_classes,n_stumps)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
    plt.title(f'Scatter plot of Predictions at {n_stumps} stumps')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Predicted Class')
    plt.show()
#########################################



class DecisionStump:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.best_dimension = None
        self.best_threshold = None
        self.best_side = None
        self.best_label = None

    def fit(self, X, y, initial_score):## a change si besoin. num_thresholds=20):
        n, m = X.shape

        best_accuracy = 0
        best_updated_score = None

        for d in range(m):
            # Récupérer les seuils uniques de la dimension
            feature_min, feature_max = X[:, d].min(), X[:, d].max()
            thresholds = np.linspace(feature_min, feature_max, num=20)
            ##corriger qu il est environ 20 threesolds

            for threshold in thresholds:
                # Essayer les deux côtés de la division
                for c in range(self.num_classes):
                    for side in ['left', 'right']:

                        # Déterminer les échantillons à mettre à jour
                        if side == 'left':
                            samples_to_update = X[:, d] <= threshold
                        else:
                            samples_to_update = X[:, d] > threshold


                        current_score = initial_score.copy()

                        one_hot = np.zeros(self.num_classes)
                        one_hot[c] = 1
                        current_score[samples_to_update] = one_hot

                        # Mettre à jour l'autre côté avec une distribution uniforme
                        current_score[~samples_to_update] = 1 / self.num_classes

                        updated_score = initial_score + current_score

                        # Calcul des prédictions
                        predictions = np.argmax(updated_score, axis=1)

                        # Calcul de la précision
                        current_accuracy = np.mean(predictions == y)

                        # Mettre à jour si meilleure précision trouvée
                        if current_accuracy > best_accuracy:
                            best_accuracy = current_accuracy
                            best_updated_score = updated_score
                            self.best_dimension = d
                            self.best_threshold = threshold
                            self.best_side = side
                            self.best_label = c

        return best_updated_score

    def predict(self, X):
        """
        Effectue des prédictions sur les nouvelles données.

        Parameters:
        - X : np.ndarray, matrice des caractéristiques (n_samples, n_features)

        Returns:
        - np.ndarray : prédictions pour chaque échantillon
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.num_classes))

        if self.best_side == 'left':
            samples_to_update = X[:, self.best_dimension] <= self.best_threshold
        else:
            samples_to_update = X[:, self.best_dimension] > self.best_threshold

        predictions[samples_to_update, self.best_label] = 1

        predictions[~samples_to_update, :] = 1 / self.num_classes

        return predictions



def train_decision_stumps(X, y, num_classes, max_stumps=25):
    n_samples, _ = X.shape
    scores = np.zeros((n_samples, num_classes))
    stumps = []

    for i in range(max_stumps):
        stump = DecisionStump(num_classes)
        stump.fit(X, y, scores)
        stumps.append(stump)
        stumps_scores = stump.predict(X)

        print("stump num", i)

        scores += stumps_scores

    return stumps

def predict_with_stumps(stumps, X_new, num_classes, model_index):
    """
    Parameters:
    - stumps : liste des stumps entraînés
    - X_new : données sur lesquelles effectuer des prédictions
    - num_classes : nombre de classes à prédire
    - model_index : index du dernier modèle à utiliser (par exemple 5 pour 5 stumps)

    Returns:
    - np.ndarray : les prédictions finales des classes (tableau d'entiers)
    """
    final_scores = np.zeros((X_new.shape[0], num_classes))

    for i in range(model_index):
        stump = stumps[i]  # Sélectionner le i-ème stump
        stump_predictions = stump.predict(X_new)  # Prédiction de ce stump
        final_scores += stump_predictions  # Ajouter ces prédictions aux scores cumulés

    predictions = np.argmax(final_scores, axis=1)

    return predictions


def boosting():
    data = pd.read_csv("train.csv")

    X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y = data.iloc[:, -1].values  # Dernière colonne (les étiquettes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    num_classes = len(np.unique(y_train))

    stumps = train_decision_stumps(X_train, y_train, num_classes=num_classes, max_stumps=25)
    for n_stumps in [1, 5, 15, 25]:
        plot_decision(X_train, y_train, stumps, n_stumps, num_classes=num_classes)


    train_accuracies = []
    test_accuracies = []

    for i in range(1, len(stumps) + 1):
        # Prédictions pour l'ensemble d'entraînement
        train_predictions = predict_with_stumps(stumps, X_train, num_classes, i)
        train_accuracy = np.mean(train_predictions == y_train)
        train_accuracies.append(train_accuracy)

        # Prédictions pour l'ensemble de test
        test_predictions = predict_with_stumps(stumps, X_test, num_classes, i)
        test_accuracy = np.mean(test_predictions == y_test)
        test_accuracies.append(test_accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 25 + 1), train_accuracies, label="train set", marker='o')
    plt.plot(range(1, 25 + 1), test_accuracies, label="test set", marker='o')
    plt.xlabel('Num of Decision Stumps')
    plt.ylabel('Accuracy')
    plt.title('Table of accuracies')
    plt.legend()
    plt.grid(True)
    plt.show()

    final_train_accuracy = train_accuracies[-1]
    final_test_accuracy = test_accuracies[-1]

    print(f"Final Train Accuracy: {final_train_accuracy:.4f}")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")










if __name__ == '__main__':
    knn_5()
    knn_anomaly()
    boosting()

