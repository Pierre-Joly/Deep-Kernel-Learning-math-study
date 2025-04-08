import torch.nn as nn
from .SGP import SGP
import torch
import numpy as np
from scipy.spatial.distance import pdist

class CNN(nn.Module):
    """
    CNN-based feature extractor for Deep Kernel Learning (DKL).

    This network transforms grayscale images of shape (N, 1, 64, 64)
    into a feature vector of shape (N, out_dim), which is then passed
    to a Gaussian Process (GP) layer.
    """
    def __init__(self, out_dim=16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),     # Output: (N, 16, 64, 64)
            nn.MaxPool2d(2),                                           # Output: (N, 16, 32, 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),    # Output: (N, 32, 32, 32)
            nn.MaxPool2d(2),                                           # Output: (N, 32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),    # Output: (N, 64, 16, 16)
            nn.MaxPool2d(2),                                           # Output: (N, 64, 8, 8)
            nn.Flatten(),                                              # Output: (N, 4096)
            nn.Linear(64 * 8 * 8, out_dim),                            # Output: (N, out_dim)
            nn.LayerNorm(out_dim),                                     # Output: (N, out_dim)
        )

        # Regression layer
        self.regression = nn.Sequential(
            nn.Linear(out_dim, 1),                                      # Output: (N, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x : Input tensor of shape (N, 1, 64, 64)

        Returns:
            Tensor of shape (N, out_dim)
        """
        x = self.encoder(x)
        x = self.regression(x)
        return x
    
    def fit(self, X_train, y_train, n_epochs=100, batch_size=32, lr=1e-3):
        """
        Train the CNN feature extractor.

        Args:
            X_train : Input training images of shape (n, 1, 64, 64)
            y_train : Training targets of shape (n,) or (n, 1)
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
        )

        for epoch in range(n_epochs):
            for X_batch, y_batch in data_loader:
                optimizer.zero_grad()
                outputs = self(X_batch).squeeze(-1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("Encoder training complete.")


class DKL(nn.Module):
    """
    Deep Kernel Learning model using a Sparse Gaussian Process (SGP).

    This model combines a NN feature extractor with a Sparse GP operating
    in the latent feature space. It supports variational inference via the
    ELBO and can compute predictive means and variances.
    """
    def __init__(self, encoder, Z_init_inducing, 
                 init_length_scale=1.0,
                 init_var=1.0,
                 init_noise=1e-2,
                 learn_inducing=True):
        super().__init__()
        self.encoder = encoder
        self.sgp = SGP(Z_init_inducing,
                 init_length_scale=1.0,
                 init_var=1.0,
                 init_noise=1e-2,
                 learn_inducing=True)

    def forward(self, X_train, y_train, X_test):
        """
        Computes predictive mean and variance at test locations.

        Args:
            X_train : Input training images of shape (n, 1, 64, 64)
            y_train : Training targets of shape (n,) or (n, 1)
            X_test  : Input test images of shape (n*, 1, 64, 64)

        Returns:
            mean     : Predictive mean at each test input (n*,)
            var_diag : Predictive marginal variance (n*,)
        """
        F_train = self.encoder(X_train)
        F_test  = self.encoder(X_test)
        return self.sgp(F_train, y_train, F_test)

    def elbo(self, X_train, y_train):
        """
        Computes the Evidence Lower Bound (ELBO) on the marginal likelihood.

        Args:
            X_train : Input training images (n, 1, 64, 64)
            y_train : Training targets

        Returns:
            elbo : Scalar ELBO value (to maximize)
        """
        F_train = self.encoder(X_train)
        return self.sgp.elbo(F_train, y_train)
