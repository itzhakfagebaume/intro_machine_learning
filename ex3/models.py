import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt



torch.manual_seed(0)

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
    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
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

class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.W = None

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.

        ########## YOUR CODE HERE ##########

        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv

        ####################################
        N, D = X.shape
        X_bias = np.hstack([X, np.ones((N, 1))])

        # Normalize X^T X and X^T Y by Ntrain
        XTX = (X_bias.T @ X_bias) / N
        XTY = (X_bias.T @ Y) / N

        # Regularization matrix (identity matrix with zero for bias term)
        identity = np.eye(D + 1)
        identity[-1, -1] = 0  # Exclude bias term from regularization

        # Compute the ridge regression weights
        self.W = np.linalg.inv(XTX + self.lambd * identity) @ XTY

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        ########## YOUR CODE HERE ##########

        # compute the predicted output of the model.
        # name your predicitons array preds.

        ####################################
        N, D = X.shape
        X_bias = np.hstack([X, np.ones((N, 1))])

        # Compute predictions using the learned weights.
        preds = X_bias @ self.W
        # transform the labels to 0s and 1s, instead of -1s and 1s.
        # You may remove this line if your code already outputs 0s and 1s.
        preds = np.where(preds >= 0, 1, -1)
        preds = (preds + 1) / 2
        return preds


def rdg_regression():
    train = pd.read_csv("train.csv")

    X_train = train.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y_train = train.iloc[:, -1].values  # Dernière colonne (les étiquettes)

    test = pd.read_csv("test.csv")

    X_test = test.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y_test = test.iloc[:, -1].values

    val = pd.read_csv("validation.csv")

    X_val = val.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y_val = val.iloc[:, -1].values

    lambdas = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []


    for lambd in lambdas:
        print(f"\nLambda: {lambd}")
        model = Ridge_Regression(lambd)

        # Fit model on training split
        model.fit(X_train, y_train)

        # Evaluate on training set
        train_predictions = model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)
        train_accuracies.append(train_accuracy)


        # Evaluate on test set
        test_predictions = model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)
        test_accuracies.append(test_accuracy)

        val_predictions = model.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        val_accuracies.append(val_accuracy)

        print(f"Training Accuracy: {train_accuracy:.2f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}")

        if lambd in [2.0,10.0]:
            plot_decision_boundaries(model, X_test, y_test)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(lambdas, val_accuracies, label='Validation Accuracy', marker='o')
    plt.plot(lambdas, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Lambda (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Accuracies vs Lambda')
    plt.legend()
    plt.grid()
    plt.show()

def gradient_des():
    learning_rate = 0.1
    iterations = 1000
    x, y = 0, 0
    path = []

    # Gradient Descent
    for i in range(iterations):
        grad_x, grad_y = 2 * (x - 3),2 * (y - 5)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        path.append((x, y))  # Stocke la position à chaque itération

    # Conversion du chemin en tableau numpy pour facilitation de la visualisation
    path = np.array(path)

    # Tracer le chemin parcouru
    plt.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), cmap='viridis', s=10)
    plt.colorbar(label="Iterations")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient Descent for f(x, y)= (x−3)**2 +(y−5)**2")
    plt.show()
    print(f"Position finale : (x, y) = ({x}, {y})")




class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):

        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########
        # define a linear operation.
        ####################################

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################
        return self.linear(x)

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x

class DummyDataset(torch.utils.data.Dataset):
    """
    Any dataset should inherit from torch.utils.data.Dataset and override the __len__ and __getitem__ methods.
    __init__ is optional.
    __len__ should return the size of the dataset.
    __getitem__ should return a tuple (data, label) for the given index.
    """

    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        self.data = torch.tensor(self.dataframe.iloc[:, :-1].values, dtype=torch.float32)
        # Charger la dernière colonne comme étiquette (label)
        self.labels = torch.tensor(self.dataframe.iloc[:, -1].values, dtype=torch.long)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def stochastic_gd():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DummyDataset(csv_file='train.csv')
    test = DummyDataset(csv_file='test.csv')
    validation = DummyDataset(csv_file='validation.csv')


    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=False)


    input_dim = train.data.shape[1]  # Nombre de caractéristiques
    output_dim = len(torch.unique(train.labels))
    val_accuracies_for_lambdas = []

    for lr in [0.1, 0.01, 0.001]:
        print(f"Training with learning rate: {lr}")
        model = Logistic_Regression(input_dim, output_dim)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        num_epochs = 10
        train_loss_values = []
        val_loss_values = []  # Pour stocker la perte sur la validation
        test_loss_values = []  # Pour stocker la perte sur le test
        val_accuracy_values = []  # Pour stocker l'exactitude sur la validation
        test_accuracy_values = []

        for epoch in range(num_epochs):
            model.train()
            loss_values = []
            ep_correct_preds = 0.

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                loss_values.append(loss.item())
                ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            mean_loss = np.mean(loss_values)
            ep_accuracy = ep_correct_preds / len(train)
            train_loss_values.append(mean_loss)



            model.eval()  # Met le modèle en mode évaluation
            val_loss = 0.0
            val_correct_preds = 0.0

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                val_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            # Calcul de la perte et de l'exactitude sur le jeu de validation
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct_preds / len(validation)
            val_loss_values.append(val_loss)
            val_accuracy_values.append(val_accuracy)

            test_loss = 0.0
            test_correct_preds = 0.0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                test_loss += loss.item()
                test_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            # Calcul de la perte et de l'exactitude sur le jeu de test
            test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct_preds / len(test)
            test_loss_values.append(test_loss)
            test_accuracy_values.append(test_accuracy)



            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mean_loss.item():.4f}, Accuracy: {ep_accuracy:.2f}')
            # print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}')
            # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}\n')
            #

        # plt.plot(val_accuracy_values, label=f'Validation Accuracy (lr={lr})')
        # plt.show()


        val_accuracies_for_lambdas.append(np.mean(val_accuracy_values))
        print(f"Validation Accuracy for lambda={lr}: {np.mean(val_accuracy_values):.4f}")
        if lr == 0.001:
            test_data = test.data.to(device).cpu().numpy()   # Convertir test.data en liste
            test_labels = test.labels.to(device).cpu().numpy()  # Convertir test.labels en liste

            # Appeler la fonction plot_decision_boundaries avec les listes
            plot_decision_boundaries(model, test_data, test_labels)

            plt.plot(range(num_epochs), train_loss_values, label='Training Loss', marker='o')
            plt.plot(range(num_epochs), val_loss_values, label='Validation Loss', marker='o')
            plt.plot(range(num_epochs), test_loss_values, label='Test Loss', marker='o')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training, Validation, and Test Loss over Epochs')
            plt.legend()
            plt.show()

def multi_class_stochastic_gd():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DummyDataset(csv_file='train_multiclass.csv')
    test = DummyDataset(csv_file='test_multiclass.csv')
    validation = DummyDataset(csv_file='validation_multiclass.csv')

    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=False)

    input_dim = train.data.shape[1]  # Nombre de caractéristiques
    output_dim = len(torch.unique(train.labels))
    val_accuracies_for_lambdas = []
    test_accuracies_for_lambdas = []

    for lr in [0.01, 0.001,0.0003]:
        model = Logistic_Regression(input_dim, output_dim)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

        num_epochs = 30
        train_loss_values = []
        train_accuracy_values = []
        val_loss_values = []  # Pour stocker la perte sur la validation
        test_loss_values = []  # Pour stocker la perte sur le test
        val_accuracy_values = []  # Pour stocker l'exactitude sur la validation
        test_accuracy_values = []

        for epoch in range(num_epochs):
            model.train()
            loss_values = []
            ep_correct_preds = 0.

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                loss_values.append(loss.item())
                ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            lr_scheduler.step()

            mean_loss = np.mean(loss_values)
            ep_accuracy = ep_correct_preds / len(train)
            train_loss_values.append(mean_loss)
            train_accuracy_values.append(ep_accuracy)

            model.eval()  # Met le modèle en mode évaluation
            val_loss = 0.0
            val_correct_preds = 0.0
            with torch.no_grad():  # Pas besoin de calculer les gradients lors de l'évaluation
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()
                    val_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            # Calcul de la perte et de l'exactitude sur le jeu de validation
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct_preds / len(validation)
            val_loss_values.append(val_loss)
            val_accuracy_values.append(val_accuracy)

            test_loss = 0.0
            test_correct_preds = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    test_loss += loss.item()
                    test_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            # Calcul de la perte et de l'exactitude sur le jeu de test
            test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct_preds / len(test)
            test_loss_values.append(test_loss)
            test_accuracy_values.append(test_accuracy)

            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mean_loss.item():.4f}, Accuracy: {ep_accuracy:.2f}')
            # print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}')
            # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}\n')
            #

        # plt.plot(val_accuracy_values, label=f'Validation Accuracy (lr={lr})')
        # plt.show()

        val_accuracies_for_lambdas.append(np.mean(val_accuracy_values))
        test_accuracies_for_lambdas.append(np.mean(test_accuracy_values))
        print(f"Validation Accuracy for lambda={lr}: {np.mean(val_accuracy_values):.4f}")

        if lr == 0.01:
            # print(test_accuracy_values)
            print("test accuracy",np.mean(test_accuracy_values))
            test_data = test.data.to(device).cpu().numpy()  # Convertir test.data en liste
            test_labels = test.labels.to(device).cpu().numpy()
            plot_decision_boundaries(model,test_data,test_labels,"stochastic gd")



    # Plotting the losses (train, validation, and test)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 2)
            plt.plot(range(num_epochs), train_loss_values, label='Train Loss')
            plt.plot(range(num_epochs), val_loss_values, label='Validation Loss')
            plt.plot(range(num_epochs), test_loss_values, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs')
            plt.legend()

            # Plotting the accuracies (train, validation, and test)
            plt.subplot(1, 2, 1)
            plt.plot(range(num_epochs), train_accuracy_values, label='Train Accuracy')
            plt.plot(range(num_epochs), val_accuracy_values, label='Validation Accuracy')
            plt.plot(range(num_epochs), test_accuracy_values, label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over Epochs')
            plt.legend()

            plt.tight_layout()
            plt.show()

    plt.plot([0.01, 0.001, 0.0003], val_accuracies_for_lambdas, label="Validation Accuracy", marker='o')
    plt.plot([0.01, 0.001, 0.0003], test_accuracies_for_lambdas, label="Test Accuracy", marker='o')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Test and Validation Accuracy vs. Learning Rate')
    plt.legend()
    plt.show()

def tree_depth(depth):
    train = pd.read_csv("train_multiclass.csv")

    X_train = train.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y_train = train.iloc[:, -1].values  # Dernière colonne (les étiquettes)

    test = pd.read_csv("test_multiclass.csv")

    X_test = test.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y_test = test.iloc[:, -1].values

    val = pd.read_csv("validation_multiclass.csv")

    X_val = val.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y_val = val.iloc[:, -1].values


    clf_depth = DecisionTreeClassifier(max_depth=depth)
    clf_depth.fit(X_train, y_train)

    train_acc_depth = accuracy_score(y_train, clf_depth.predict(X_train))
    val_acc_depth = accuracy_score(y_val, clf_depth.predict(X_val))
    test_acc_depth = accuracy_score(y_test, clf_depth.predict(X_test))

    plt.figure(figsize=(12, 5))
    plot_decision_boundaries(clf_depth, X_test, y_test,'Decision Tree')
    print(
        f"Decision Tree (max_depth={depth}) - \n Train Accuracy: {train_acc_depth:.4f},"
        f"\n Validation Accuracy: {val_acc_depth:.4f},\n Test Accuracy: {test_acc_depth:.4f}")


def multi_class_stochastic_gd_with_ridge():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DummyDataset(csv_file='train_multiclass.csv')
    test = DummyDataset(csv_file='test_multiclass.csv')
    validation = DummyDataset(csv_file='validation_multiclass.csv')

    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=False)

    input_dim = train.data.shape[1]  # Nombre de caractéristiques
    output_dim = len(torch.unique(train.labels))

    lambdas = [0., 2., 4., 6., 8., 10.]  # Valeurs de régularisation L2 à tester
    best_lambda = None
    best_val_acc = 0.0
    val_accuracies_for_lambdas = []
    test_accuracies_for_lambdas = []

    for lam in lambdas:
        print(f"\n=== Training with ridge regularization (lambda={lam}) ===")
        model = Logistic_Regression(input_dim, output_dim)
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        # Ajout de la régularisation L2 à l'optimiseur
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=lam)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

        num_epochs = 30
        train_loss_values = []
        train_accuracy_values = []
        val_loss_values = []  # Pour stocker la perte sur la validation
        test_loss_values = []  # Pour stocker la perte sur le test
        val_accuracy_values = []  # Pour stocker l'exactitude sur la validation
        test_accuracy_values = []

        for epoch in range(num_epochs):
            model.train()
            loss_values = []
            ep_correct_preds = 0.

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                loss_values.append(loss.item())
                ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            lr_scheduler.step()

            mean_loss = np.mean(loss_values)
            ep_accuracy = ep_correct_preds / len(train)
            train_loss_values.append(mean_loss)
            train_accuracy_values.append(ep_accuracy)

            model.eval()  # Met le modèle en mode évaluation
            val_loss = 0.0
            val_correct_preds = 0.0
            with torch.no_grad():  # Pas besoin de calculer les gradients lors de l'évaluation
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()
                    val_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            # Calcul de la perte et de l'exactitude sur le jeu de validation
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct_preds / len(validation)
            val_loss_values.append(val_loss)
            val_accuracy_values.append(val_accuracy)

            test_loss = 0.0
            test_correct_preds = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    test_loss += loss.item()
                    test_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

            # Calcul de la perte et de l'exactitude sur le jeu de test
            test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct_preds / len(test)
            test_loss_values.append(test_loss)
            test_accuracy_values.append(test_accuracy)

        # Enregistrer les meilleurs résultats pour chaque valeur de lambda
        val_accuracies_for_lambdas.append(val_accuracy_values[-1])
        test_accuracies_for_lambdas.append(test_accuracy_values[-1])

        if val_accuracy_values[-1] > best_val_acc:
            best_val_acc = val_accuracy_values[-1]
            best_lambda = lam

        if lam == 0:
            # Visualisation des frontières de décision pour le meilleur modèle
            test_data = test.data.to(device).cpu().numpy()  # Convertir test.data en liste
            test_labels = test.labels.to(device).cpu().numpy()
            plot_decision_boundaries(model, test_data, test_labels,
                                     title=f"Ridge Regularization (lambda={best_lambda})")

    print(f"\nBest lambda: {best_lambda} with validation accuracy: {best_val_acc:.4f}")

    # Afficher les résultats de précision pour chaque valeur de lambda
    for lam, val_acc, test_acc in zip(lambdas, val_accuracies_for_lambdas, test_accuracies_for_lambdas):
        print(f"Lambda = {lam}, Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")



if __name__ == '__main__':
    #rdg_regression()
    #gradient_des()
    #stochastic_gd()
    # multi_class_stochastic_gd()
    # tree_depth(2)
    # tree_depth(10)
    multi_class_stochastic_gd_with_ridge()


