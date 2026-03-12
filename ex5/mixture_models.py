from math import log
import torch
import torch.nn as nn
from dataset import EuropeDataset
import matplotlib.pyplot as plt
import pandas as pd
def normalize_tensor(tensor, d):
    """
    Normalize the input tensor along the specified axis to have a mean of 0 and a std of 1.
    
    Parameters:
        tensor (torch.Tensor): Input tensor to normalize.
        d (int): Axis along which to normalize.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = torch.mean(tensor, dim=d, keepdim=True)
    std = torch.std(tensor, dim=d, keepdim=True)
    normalized = (tensor - mean) / std
    return normalized


class GMM(nn.Module):
    def __init__(self, n_components, init_means=None):
        """
        Gaussian Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of Gaussian components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        if init_means is not None:
            self.means = nn.Parameter(init_means.clone())
        else:
            # Means of the Gaussian components (n_components x 2 for 2D data)
            self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of the variance of the Gaussian components (n_components x 2 for 2D data)
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # Log-variances (diagonal covariance)




    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """        
        #### YOUR CODE GOES HERE ####
        log_weights = torch.nn.functional.log_softmax(self.weights, dim=0)

        # Step 2: Reshape inputs for broadcasting
        X = X.unsqueeze(1)  # Shape: (n_samples, 1, 2) to match (n_components, 2)
        means = self.means.unsqueeze(0)  # Shape: (1, n_components, 2)
        variances = torch.exp(self.log_variances).unsqueeze(0)  # Shape: (1, n_components, 2)

        # Step 3: Compute log p(x|k) for each component
        log_variances = torch.log(variances)  # Shape: (1, n_components, 2)
        diff = X - means  # Difference between data and means, shape: (n_samples, n_components, 2)
        log_const = -0.5 * log(2 * torch.pi)
        log_p_x_given_k = (
                log_const  # Constant term
                - 0.5 * log_variances.sum(dim=2)  # - log(σ1^2) - log(σ2^2)
                - 0.5 * ((diff ** 2) / variances).sum(dim=2)  # - [(x1 - μ1)^2 / σ1^2 + (x2 - μ2)^2 / σ2^2]
        )  # Shape: (n_samples, n_components)

        # Step 4: Combine log p(x|k) and log p(k)
        log_joint = log_p_x_given_k + log_weights  # Shape: (n_samples, n_components)

        # Step 5: Compute log p(x) using log-sum-exp
        log_likelihood = torch.logsumexp(log_joint, dim=1)  # Shape: (n_samples,)

        return log_likelihood

    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        #### YOUR CODE GOES HERE ####
        loss = -torch.mean(log_likelihood)

        return loss


    def sample(self, n_samples):
        """
        Generate samples from the GMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        probs = torch.nn.functional.softmax(self.weights, dim=0)

        # Step 2: Sample component indices
        component_indices = torch.multinomial(probs, n_samples, replacement=True)

        # Step 3: Sample from each selected Gaussian component
        samples = []
        for k in component_indices:
            # Mean and std for the chosen component
            mean = self.means[k]
            std = torch.exp(0.5 * self.log_variances[k])

            # Sample from N(mean, std)
            z = torch.randn(2)  # Standard Normal noise
            x = mean + z * std  # Transform to desired Gaussian
            samples.append(x)

        return torch.stack(samples)
    
    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        mean = self.means[label]
        std = torch.exp(0.5 * self.log_variances[label])

        # Sample n points from N(mean, std)
        z = torch.randn(n_samples, 2)  # Standard Normal noise
        x = mean + z * std  # Transform to desired Gaussian
        return x



class UMM(nn.Module):
    def __init__(self, n_components, init_means=None):
        """
        Uniform Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of uniform components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Center value of the uniform components (n_components x 2 for 2D data)
        if init_means is not None:
            self.means = nn.Parameter(init_means.clone())
        else:
            # Means of the Gaussian components (n_components x 2 for 2D data)
            self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of size of the uniform components (n_components x 2 for 2D data)
        self.log_sizes = nn.Parameter(torch.log(torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))


    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """
        #### YOUR CODE GOES HERE ####
        #### Compute log weights (log p(k)) ####
        log_weights = torch.nn.functional.log_softmax(self.weights, dim=0)  # Shape: (n_components,)

        #### Reshape inputs for broadcasting ####
        X = X.unsqueeze(1)  # Shape: (n_samples, 1, 2)
        means = self.means.unsqueeze(0)  # Shape: (1, n_components, 2)
        scales = torch.exp(self.log_sizes).unsqueeze(0)  # Shape: (1, n_components, 2)

        #### Compute log p(x|k) for each component ####
        lower_bounds = means - scales / 2 # cx - s1, cy - s2 for each component
        upper_bounds = means + scales / 2  # cx + s1, cy + s2 for each component

        # Check if X lies within the bounds
        within_bounds = (X >= lower_bounds) & (X <= upper_bounds)  # Shape : (n_samples, n_components, 2)
        within_bounds = within_bounds.all(dim=2)  # Shape : (n_samples, n_components)

        # Compute log p(x|k)
        log_p_x_given_k = torch.where(
            within_bounds,
            -torch.log(scales[:, :, 0]) - torch.log(scales[:, :, 1]),  # -log(s1 * s2)
            torch.tensor(-1e6, device=X.device)  # Assign a large negative number for out-of-bounds
        )  # Shape : (n_samples, n_components)

        #### Combine log p(x|k) and log p(k) to get log joint ####
        log_joint = log_p_x_given_k + log_weights  # Shape: (n_samples, n_components)

        #### Compute log p(x) using log-sum-exp ####
        log_likelihood = torch.logsumexp(log_joint, dim=1)  # Shape: (n_samples,)

        return log_likelihood


        
    
    
    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        #### YOUR CODE GOES HERE ####
        loss = -torch.mean(log_likelihood)

        return loss


    def sample(self, n_samples):
        """
        Generate samples from the UMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        probs = torch.nn.functional.softmax(self.weights, dim=0)

        # Step 2: Sample component indices
        component_indices = torch.multinomial(probs, n_samples, replacement=True)

        # Step 3: Sample from each selected Gaussian component
        samples = []
        for k in component_indices:
            # Mean and std for the chosen component
            center = self.means[k]
            scale = torch.exp(self.log_sizes[k])  # Ensure positivity

            # Define the bounds for the uniform distribution
            lower_bound = center - scale / 2
            upper_bound = center + scale / 2

            # Create a uniform distribution and sample
            uniform_dist = torch.distributions.Uniform(lower_bound, upper_bound)
            x = uniform_dist.sample()  # Generate one sample
            samples.append(x)

        return torch.stack(samples)

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        center = self.means[label]
        scale = torch.exp(self.log_sizes[label])

        lower_bound = center - scale / 2
        upper_bound = center + scale / 2

        # Create a uniform distribution
        uniform_dist = torch.distributions.Uniform(lower_bound, upper_bound)

        # Sample n points from this uniform distribution
        samples = uniform_dist.sample((n_samples,))
        return samples

def questions_1():
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    batch_size = 4096
    num_epochs = 50
    # Use Adam optimizer
    # TODO: Determine learning rate
    # learning_rate for GMM = 0.01
    # learning_rate for UMM = 0.001

    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #### YOUR CODE GOES HERE ####

    n_components = [1, 5, 10, len(train_dataset.labels.unique())]
    for n_component in n_components:
        print(f"Training GMM with n_components = {n_component}")
        gmm = UMM(n_components=n_component)

        # Optimizer and learning rate
        learning_rate = 0.001
        optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                X = batch[0]  # Assuming your dataset outputs features
                optimizer.zero_grad()

                # Compute negative log-likelihood
                log_likelihood = gmm.forward(X)  # Log-likelihood for all samples in the batch
                loss = -log_likelihood.mean()  # Negative log-likelihood (minimize this)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Generate 1000 samples from the trained GMM
        samples = gmm.sample(1000)

        # Plot the samples
        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy(), alpha=0.6, s=10)
        plt.title("Scatter Plot of 1000 Samples from the GMM")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.show()

        all_samples = []
        colors = []
        for k in range(n_component):
            # Conditional samples for the k-th Gaussian
            component_samples = gmm.conditional_sample(100, k)
            all_samples.append(component_samples)
            colors.extend([k] * 100)  # Color code for the component

        # Combine all samples
        all_samples = torch.cat(all_samples, dim=0).detach().numpy()

        # Plot the samples
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(all_samples[:, 0], all_samples[:, 1], c=colors, cmap='tab10', alpha=0.7, s=10)
        plt.title(f"Scatter Plot of 100 Samples per Component (n_components={n_component})")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(scatter, label="Gaussian Component")
        plt.grid(True)
        plt.show()

def questions_2():
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    batch_size = 4096
    num_epochs = 50
    # Use Adam optimizer
    # TODO: Determine learning rate
    # learning_rate for GMM = 0.01
    # learning_rate for UMM = 0.001

    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #### YOUR CODE GOES HERE ####

    train_features = train_dataset.features.numpy()  # Convertir les features en numpy array
    train_labels = train_dataset.labels.numpy()  # Convertir les labels en numpy array

    # Créer un DataFrame pandas
    train_df = pd.DataFrame({
        'latitude': train_features[:, 0],  # Première colonne (normalisée)
        'longitude': train_features[:, 1],  # Deuxième colonne (normalisée)
        'country': train_labels  # Labels (pays)
    })

    country_means = train_df.groupby('country')[['latitude', 'longitude']].mean()

    # Convertir en un tensor PyTorch pour les passer au GMM
    country_means_tensor = torch.tensor(country_means.values, dtype=torch.float32)

    n_component = len(train_dataset.labels.unique())
    print(f"Training GMM with n_components = {n_component}")
    gmm = UMM(n_components=n_component, init_means=country_means_tensor)
    # gmm = UMM(n_components=n_component)

    # Optimizer and learning rate
    learning_rate = 0.001
    optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate)
    train_log_likelihoods = []
    test_log_likelihoods = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_train_log_likelihood = 0.0  # Pour accumuler le log-likelihood d'entraînement
        total_test_log_likelihood = 0.0

        for batch in train_loader:
            X = batch[0]  # Assuming your dataset outputs features
            optimizer.zero_grad()

            # Compute negative log-likelihood
            log_likelihood = gmm.forward(X)  # Log-likelihood for all samples in the batch
            total_train_log_likelihood += log_likelihood.sum().item()
            loss = -log_likelihood.mean()  # Negative log-likelihood (minimize this)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        with torch.no_grad():  # Pas besoin de rétropropager pour l'ensemble de test
            for batch in test_loader:
                X_test = batch[0]  # Supposons que batch[0] contient les caractéristiques de test
                log_likelihood_test = gmm.forward(X_test)
                total_test_log_likelihood += log_likelihood_test.sum().item()

        train_log_likelihoods.append(total_train_log_likelihood / len(train_loader.dataset))
        test_log_likelihoods.append(total_test_log_likelihood / len(test_loader.dataset))

        if epoch+1 in [1, 10, 20, 30, 40, 50]:

            # Generate 1000 samples from the trained GMM
            samples = gmm.sample(1000)

            # Plot the samples
            plt.figure(figsize=(8, 6))
            plt.scatter(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy(), alpha=0.6, s=10)
            plt.title("Scatter Plot of 1000 Samples from the GMM {}".format(epoch + 1))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.show()

            all_samples = []
            colors = []
            for k in range(n_component):
                # Conditional samples for the k-th Gaussian
                component_samples = gmm.conditional_sample(100, k)
                all_samples.append(component_samples)
                colors.extend([k] * 100)  # Color code for the component

            # Combine all samples
            all_samples = torch.cat(all_samples, dim=0).detach().numpy()

            # Plot the samples
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(all_samples[:, 0], all_samples[:, 1], c=colors, cmap='tab10', alpha=0.7, s=10)
            plt.title("Scatter Plot of 100 Samples per Component {}".format(epoch + 1))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.colorbar(scatter, label="Gaussian Component")
            plt.grid(True)
            plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_log_likelihoods, label='Training Log Likelihood', color='blue')
    plt.plot(range(1, num_epochs + 1), test_log_likelihoods, label='Testing Log Likelihood', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Log Likelihood')
    plt.title('Training and Testing Mean Log Likelihood vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage
if __name__ == "__main__":
    
    # torch.manual_seed(42)
    # train_dataset = EuropeDataset('train.csv')
    # test_dataset = EuropeDataset('test.csv')
    #
    # batch_size = 4096
    # num_epochs = 50
    # # Use Adam optimizer
    # # TODO: Determine learning rate
    # # learning_rate for GMM = 0.01
    # # learning_rate for UMM = 0.001
    #
    # train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # #### YOUR CODE GOES HERE ####
    #questions_1()
    questions_2()

