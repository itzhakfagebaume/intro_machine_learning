import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()

        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        logits = self.logistic_regression(features)
        return logits


def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    ### YOUR CODE HERE ###
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device).float()

            # logits de sortie
            logits = model(imgs)

            # Appliquer la fonction Sigmoid pour obtenir des probabilités
            probabilities = torch.sigmoid(logits)

            # Convertir les probabilités en prédictions binaires (0 ou 1)
            predictions = (probabilities >= 0.5).float()

            # Compter les prédictions correctes
            correct_predictions += (predictions.squeeze() == labels).sum().item()
            total_predictions += labels.size(0)

    # Calculer la précision
    accuracy = correct_predictions / total_predictions
    return accuracy


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        ### YOUR CODE HERE ###
        imgs = imgs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        return running_loss / len(train_loader)


# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
# model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32
num_of_epochs = 1
learning_rate = learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
path = '/content/drive/My Drive/whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")

    # Optimiseur avec le taux d'apprentissage actuel
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ### Train the model

    patience = 5  # Nombre d'époques avant d'arrêter si pas d'amélioration
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch

        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        # Stopping condition
        ### YOUR CODE HERE ###
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0  # Réinitialiser le compteur si l'amélioration est détectée
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered. No improvement for {patience} epochs.')
            break

    # Compute the test accuracy
    test_acc = compute_accuracy(model, test_loader, device)
    print("test accuracy",test_acc,f" for learning rate: {lr}\n")


# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
# model = ResNet18(pretrained=False, probing=False)
# Linear probing

learning_rate = learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = ResNet18(pretrained=True, probing=True)
    # Fine-tuning
    # model = ResNet18(pretrained=True, probing=False)

    transform = model.transform
    batch_size = 32
    num_of_epochs = 1

    path = '/content/drive/My Drive/whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ### Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimiseur avec le taux d'apprentissage actuel
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ### Train the model

    patience = 5  # Nombre d'époques avant d'arrêter si pas d'amélioration
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch

        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        # Stopping condition
        ### YOUR CODE HERE ###
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0  # Réinitialiser le compteur si l'amélioration est détectée
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered. No improvement for {patience} epochs.')
            break

    # Compute the test accuracy
    test_acc = compute_accuracy(model, test_loader, device)
    print("test accuracy",test_acc,f" for learning rate: {lr}\n")




# Liste pour stocker les échantillons
misclassified_samples = []
misclassified_labels = []
misclassified_images = []



import matplotlib.pyplot as plt
# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
# model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
model = ResNet18(pretrained=True, probing=False)

learning_rate = learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = ResNet18(pretrained=True, probing=False)

    transform = model.transform
    batch_size = 32
    num_of_epochs = 1

    path = '/content/drive/My Drive/whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ### Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimiseur avec le taux d'apprentissage actuel
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ### Train the model



    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch

        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        # Stopping condition
        if lr == learning_rates[0]:
            worst_model_state = model.state_dict()
            worst_model = model
        if lr == learning_rates[3]:
            best_model_state = model.state_dict()
            best_model = model



    # Compute the test accuracy
    test_acc = compute_accuracy(model, test_loader, device)
    print("test accuracy",test_acc,f" for learning rate: {lr}\n")


with torch.no_grad():
    # Passer les modèles en mode évaluation
    best_model.eval()
    worst_model.eval()

    misclassified_samples = []
    misclassified_labels = []
    misclassified_images = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Prédictions du pire modèle
        logits_worst = worst_model(images)
        probs_worst = torch.sigmoid(logits_worst)
        worst_pred = (probs_worst > 0.5).float()

        # Prédictions du meilleur modèle
        logits_best = best_model(images)
        probs_best = torch.sigmoid(logits_best)
        best_pred = (probs_best > 0.5).float()

        # Comparer les prédictions
        for i in range(len(labels)):
            if worst_pred[i].item() != best_pred[i].item() and best_pred[i].item() == labels[i].item():
                misclassified_samples.append(images[i].cpu())
                misclassified_labels.append(labels[i].item())
                misclassified_images.append(images[i].cpu())

                # Stopper dès que 5 exemples sont trouvés
                if len(misclassified_samples) >= 5:
                    break
        if len(misclassified_samples) >= 5:
            break

# Visualiser les échantillons mal classifiés
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    image = misclassified_images[i].permute(1, 2, 0)  # Réorganiser les dimensions pour l'affichage
    label = misclassified_labels[i]
    axes[i].imshow(image.numpy())
    axes[i].axis('off')
    axes[i].set_title(f"Label: {label}")
plt.tight_layout()
plt.show()






