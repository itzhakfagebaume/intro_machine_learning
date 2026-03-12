from datetime import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from helpers import *
import pandas as pd

class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        #### YOUR CODE HERE ####
        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long

        #### END OF YOUR CODE ####
        self.dataframe = pd.read_csv(csv_file)
        self.data = torch.tensor(self.dataframe.iloc[:, 1:-1].values, dtype=torch.float32)
        # Charger la dernière colonne comme étiquette (label)
        self.labels = torch.tensor(self.dataframe.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        return self.data[idx], self.labels[idx]
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        #### YOUR CODE HERE ####
        self.layers = []

        # Input layer (from input_dim to hidden_dim)
        self.layers.append(nn.Linear(2, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())  # Adding activation function

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())  # Adding activation function

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Use nn.Sequential to stack all layers
        self.model = nn.Sequential(*(self.layers))

    def forward(self, x):
        #### YOUR CODE HERE ####
        return self.model(x)


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):    

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)    
    
    #### YOUR CODE HERE ####
    # initialize your criterion and optimizer here
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        #### YOUR CODE HERE ####
        # perform training epoch here
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in trainloader:
            # Envoi des données sur le bon appareil (GPU/CPU)
            #inputs, labels = inputs.to(model.device), labels.to(model.device)

            optimizer.zero_grad()  # Réinitialisation des gradients
            outputs = model(inputs)  # Passer les données dans le modèle
            loss = criterion(outputs, labels)  # Calcul de la perte

            loss.backward()  # Rétropropagation des gradients
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()

            # Calcul de la précision
            _, predicted = torch.max(outputs, 1)  # Prédictions du modèle
            correct_train += (predicted == labels).sum().item()
            total_train += inputs.size(0)  # Ajouter le nombre d'exemples traités dans ce batch

        train_acc = correct_train / total_train
        train_losses.append(running_loss / len(trainloader))
        train_accs.append(train_acc)


        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valloader:
                #inputs, labels = inputs.to(model.device), labels.to(model.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_losses.append(val_loss / len(valloader))
        val_accs.append(val_acc)

        # Test
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
               # inputs, labels = inputs.to(model.device), labels.to(model.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test
        test_losses.append(test_loss / len(testloader))
        test_accs.append(test_acc)

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1],
                                                                                       val_accs[-1], test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses

def exemple_or_norm():
    torch.manual_seed(0)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    output_dim = len(train_dataset.labels.unique())
    model = MLP(6, 16, output_dim)

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset,
                                                                                          val_dataset, test_dataset,
                                                                                          model, lr=0.001,
                                                                                          epochs=50, batch_size=256)

    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.legend()
    plt.show()

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Decision Boundaries', implicit_repr=False)


def learning_rate():
    torch.manual_seed(0)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    # Trouver le nombre de classes
    output_dim = len(train_dataset.labels.unique())

    # Différents taux d'apprentissage
    learning_rates = [1.0, 0.01, 0.001, 0.00001]
    val_losses_dict = {}  # Stocke les pertes de validation pour chaque learning rate

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")

        # Initialisation du modèle
        model = MLP(6, 16, output_dim)

        # Entraînement
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(
            train_dataset, val_dataset, test_dataset, model, lr=lr, epochs=50, batch_size=256
        )

        # Stocker les pertes de validation pour ce taux d'apprentissage
        val_losses_dict[lr] = val_losses

    # Tracer les pertes de validation pour chaque learning rate
    plt.figure(figsize=(10, 6))
    for lr, losses in val_losses_dict.items():
        plt.plot(losses, label=f"LR={lr}")
    plt.title("Validation Loss vs Epochs for Different Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Analyse des taux d'apprentissage
    # print("Analysis:")
    # print(
    #     "- Un learning rate trop élevé (e.g., 1.0) peut entraîner une instabilité. Les pertes oscillent ou ne convergent pas.")
    # print(
    #     "- Un learning rate trop faible (e.g., 0.00001) ralentit la convergence. Les pertes diminuent très lentement.")
    # print(
    #     "- Un learning rate intermédiaire (e.g., 0.01 ou 0.001) est souvent optimal : les pertes diminuent régulièrement.")

def epochs_experiment():
    torch.manual_seed(0)

    # Chargement des datasets
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    # Trouver le nombre de classes
    output_dim = len(train_dataset.labels.unique())

    # Initialisation du modèle
    model = MLP(6, 16, output_dim)

    # Entraînement sur 100 époques
    epochs = 100
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(
        train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=epochs, batch_size=256
    )

    # Tracer les pertes
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(val_losses, label='Validation Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='green')
    plt.title("Loss over Train, Validation, and Test Sets")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Analyse du nombre d'époques
    # print("Analysis:")
    # print(
    #     "- Avec trop peu d'époques : le modèle n'a pas le temps de bien s'entraîner, et les pertes restent élevées.")
    # print("- Avec un nombre optimal d'époques : les pertes diminuent et atteignent un plateau.")
    # print(
    #     "- Avec trop d'époques : le modèle peut sur-apprendre (overfitting), ce qui entraîne une augmentation de la perte de validation et de test.")


def batch_size_experiment():
    torch.manual_seed(0)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    output_dim = len(train_dataset.labels.unique())  # Number of classes
    results = {}

    # Configuration des tailles de batch et des époques correspondantes
    batch_sizes = [1, 16, 128, 1024]


    epochs_list = [1, 10, 50, 50]

    for batch_size, epochs in zip(batch_sizes, epochs_list):
        print(f"Training with batch size {batch_size} and {epochs} epochs...")

        # Initialisation du modèle et des structures de stockage des résultats
        model = MLP(6, 16, output_dim)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        val_accs = []
        batch_losses = []

        for ep in range(epochs):
            # Mesurer le temps pour une époque

            # Entraînement
            model.train()
            running_loss = 0.0
            for inputs, targets in trainloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_losses.append(loss.item())

            train_losses.append(running_loss / len(trainloader))

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in valloader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            val_accs.append(correct / total)

            print(f"Epoch {ep+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

        # Stocker les résultats pour ce batch size
        results[batch_size] = {
            'train_losses': train_losses,
            'val_accs': val_accs,
            'batch_losses': batch_losses,
            'iterations_per_epoch': len(trainloader)
        }

    # Graphique pour la précision de validation
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        plt.plot(results[batch_size]['val_accs'], label=f"Batch Size {batch_size}")
    plt.title("Validation Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    #speed train
    for batch_size in batch_sizes:
        dataset_size = len(train_dataset)
        iterations_per_epoch = dataset_size // batch_size
        print(f"Batch size: {batch_size}, Iterations per epoch: {iterations_per_epoch}")

    # Graphique pour la perte d'entraînement (stabilité)
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        batch_losses = results[batch_size]['batch_losses']
        plt.plot(batch_losses, label=f"Batch Size {batch_size}")

    plt.title("Training Loss vs. Batch")
    plt.xlabel("Batch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

def six_mlp():
    torch.manual_seed(0)


    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    output_dim = len(train_dataset.labels.unique())

    dephts = [1,2,6,10,6,6,6]
    widhts = [16,16,16,16,8,32,64]

    train_accuracies_16 = []
    val_accuracies_16 = []
    test_accuracies_16 = []

    train_accuracies_6 = []
    val_accuracies_6 = []
    test_accuracies_6 = []

    for depht, widht in zip(dephts, widhts):
        print(f"Training with depht {depht} and {widht} widht...")

        model = MLP(depht, widht, output_dim)

        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset,
                                                                                          val_dataset, test_dataset,
                                                                                          model, lr=0.001,
                                                                                          epochs=50, batch_size=256)
        if widht == 16:
            train_accuracies_16.append(train_accs[-1])
            val_accuracies_16.append(val_accs[-1])
            test_accuracies_16.append(test_accs[-1])

        if depht == 6:
            train_accuracies_6.append(train_accs[-1])
            val_accuracies_6.append(val_accs[-1])
            test_accuracies_6.append(test_accs[-1])

        print(f"validation accuracy for depht {depht} and {widht} widht:", val_accs[-1])

        if (depht == 10 and widht == 16) or (depht == 1 and widht == 16):

            plt.figure()
            plt.plot(train_losses, label='Train', color='red')
            plt.plot(val_losses, label='Val', color='blue')
            plt.plot(test_losses, label='Test', color='green')
            plt.title('Losses')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(train_accs, label='Train', color='red')
            plt.plot(val_accs, label='Val', color='blue')
            plt.plot(test_accs, label='Test', color='green')
            plt.title('Accs.')
            plt.legend()
            plt.show()

            train_data = pd.read_csv('train.csv')
            val_data = pd.read_csv('validation.csv')
            test_data = pd.read_csv('test.csv')
            plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values,
                                     f'Decision Boundaries {depht} , {widht}', implicit_repr=False)

    plt.figure(figsize=(10, 6))
    plt.plot([1,2,6,10], train_accuracies_16, label="Train Accuracy", marker="o", color="red")
    plt.plot([1,2,6,10], val_accuracies_16, label="Validation Accuracy", marker="o", color="blue")
    plt.plot([1,2,6,10], test_accuracies_16, label="Test Accuracy", marker="o", color="green")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Accuracy")
    plt.title("Effect of Depth on MLP Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot([16,8,32,64], train_accuracies_6, label="Train Accuracy", marker="o", color="red")
    plt.plot([16,8,32,64], val_accuracies_6, label="Validation Accuracy", marker="o", color="blue")
    plt.plot([16,8,32,64], test_accuracies_6, label="Test Accuracy", marker="o", color="green")
    plt.xlabel("Width of Network")
    plt.ylabel("Accuracy")
    plt.title("Effect of Width on MLP Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

def hundred_layer():
    torch.manual_seed(0)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    output_dim = len(train_dataset.labels.unique())
    model = MLP(100, 4, output_dim)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    #### YOUR CODE HERE ####
    # initialize your criterion and optimizer here
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    grad_tracking_layers = [0, 30*3, 60*3, 90*3, 95*3, 99*3]
    grad_magnitudes = {step: [] for step in grad_tracking_layers}

    # For each epoch, track gradient magnitudes for specified layers
    for ep in range(10):
        model.train()
        epoch_grad_magnitudes = {step: [] for step in grad_tracking_layers}  # Reset for each epoch

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Compute gradients

            # Track gradient magnitudes for specified layers (not for every batch_idx)
            for layer_idx in grad_tracking_layers:
                layer_grad = None
                # Find the gradient for the specific layer by indexing through the model
                # Assuming model is a sequential model (nn.Sequential)
                if model.layers[layer_idx].weight.grad is not None:
                    layer_grad = torch.norm(model.layers[layer_idx].weight.grad).item() **2    # ||grad||2 of the weight in that layer
                if layer_grad is not None:
                    epoch_grad_magnitudes[layer_idx].append(layer_grad)  # Store gradient magnitude for the layer

            optimizer.step()  # Update model weights

        # Aggregate gradient magnitudes for the epoch and store for later use (tracking per epoch)
        for layer_idx in grad_tracking_layers:
            if epoch_grad_magnitudes[layer_idx]:
                avg_grad_magnitude = sum(epoch_grad_magnitudes[layer_idx]) / len(
                    epoch_grad_magnitudes[layer_idx])  # Average magnitude
                grad_magnitudes[layer_idx].append(avg_grad_magnitude)

    plt.figure(figsize=(10, 6))
    for step in grad_tracking_layers:
        print(grad_magnitudes[step])
        plt.plot(range(10), grad_magnitudes[step], label=f"Step {step/2}")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Gradient Magnitude")
    plt.title("Mean Gradient Magnitude Across Epochs")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()











if __name__ == '__main__':
    # seed for reproducibility
    #changer le code si je veux rajouter une normalisation
    #exemple_or_norm()
    #learning_rate()
    #epochs_experiment()
    #batch_size_experiment()
    #six_mlp()
    hundred_layer()

