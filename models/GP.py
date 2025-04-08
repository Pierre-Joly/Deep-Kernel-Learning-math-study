import torch
import math

class GaussianProcess(torch.nn.Module):
    """
    Gaussian Process Regression (Full GP) using RBF kernel.

    Predictive equations (no inducing points):
        mean(x*) = \mu + k(x*, X) K^-1 (y - \mu)
        cov(x*)  = k(x*, x*) - k(x*, X) K^-1 k(X, x*)

    Log marginal likelihood:
        log p(y | X) = -0.5 (y - \mu)^T K^-1 (y - \mu)
                      -0.5 log|K| - n/2 log(2 \pi)

    Hyperparameters:
      - log_length_scale : log l, characteristic length-scale of the kernel
      - log_signal_variance : log \sigma_f^2, variance of the RBF kernel
      - log_noise : log \sigma_n^2, measurement noise
      - mean : mean function value (scalar constant)

    """

    def __init__(self, initial_length_scale=1.0, initial_variance=1.0, initial_noise=1e-2, initial_mean=0.0):
        super().__init__()
        # Log-transformed parameters for positivity constraint
        self.log_length_scale = torch.nn.Parameter(torch.log(torch.tensor(initial_length_scale)))
        self.log_signal_variance = torch.nn.Parameter(torch.log(torch.tensor(initial_variance)))
        self.log_noise = torch.nn.Parameter(torch.log(torch.tensor(initial_noise)))

        # Constant mean function
        self.mean = torch.nn.Parameter(torch.tensor(initial_mean))

    @property
    def length_scale(self):
        return torch.exp(self.log_length_scale)

    @property
    def signal_variance(self):
        return torch.exp(self.log_signal_variance)

    @property
    def noise_variance(self):
        return torch.exp(self.log_noise)
    
    def rbf_kernel(self, X1, X2):
        """
        Computes the Radial Basis Function (RBF) kernel matrix between two sets of inputs.

        k(x1, x2) = signal_var * exp(-0.5 * ||x1 - x2||^2 / length_scale^2)

        Args:
            X1: Tensor of shape (n1, d)
            X2: Tensor of shape (n2, d)

        Returns:
            A kernel matrix of shape (n1, n2) with entries k(X1[i], X2[j]).
        """
        X1_sq = (X1**2).sum(dim=1, keepdim=True)        # shape (n1, 1)
        X2_sq = (X2**2).sum(dim=1, keepdim=True)        # shape (n2, 1)
        
        dist_sq = X1_sq - 2 * X1 @ X2.t() + X2_sq.t()    # shape (n1, n2)
        
        return self.signal_variance * torch.exp(-0.5 * dist_sq / (self.length_scale**2))

    def _compute_Knn(self, X_train):
        """
        Computes the full covariance matrix with added noise and jitter for stability.

        K = k(X, X) + (\sigma_n^2 + \epsilon) * I

        Args:
            X_train: (n, d) training inputs

        Returns:
            (n, n) covariance matrix
        """
        K = self.rbf_kernel(X_train, X_train)
        jitter = 1e-6
        noise_term = (self.noise_variance + jitter) * torch.eye(X_train.shape[0], device=X_train.device)
        return K + noise_term

    def forward(self, X_train, y_train, X_test):
        """
        Computes predictive mean and covariance matrix for test inputs.

        Args:
            X_train: (n, d) training inputs
            y_train: (n,) or (n, 1) training targets
            X_test:  (n*, d) test inputs

        Returns:
            mean: predictive mean at test inputs, shape (n*)
            cov:  predictive covariance matrix, shape (n*, n*)
        """
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(-1)

        # Compute training kernel matrix with noise
        K = self._compute_Knn(X_train)
        L = torch.linalg.cholesky(K)

        # Center training outputs
        y_centered = y_train - self.mean

        # Compute cross- and self-covariance for test points
        K_s = self.rbf_kernel(X_train, X_test)
        K_ss = self.rbf_kernel(X_test, X_test)

        # Solve K alpha = y
        alpha = torch.cholesky_solve(y_centered, L)

        # Predictive mean
        mean = self.mean + K_s.t().matmul(alpha).squeeze(-1)

        # Predictive variance
        v = torch.linalg.solve(L, K_s)

        K_ss_diag = self.rbf_kernel(X_test, X_test).diagonal()

        var = K_ss_diag - (v * v).sum(dim=0) + self.noise_variance
        
        return mean, var
    
    def log_marginal_likelihood(self, X_train, y_train):
        """
        Computes the log marginal likelihood of the model.

        log p(y | X) = -0.5 * (y - \mu)^T K^-1 (y - \mu)
                     -0.5 * log det(K)
                     -0.5 * n log(2 \pi)

        Args:
            X_train: (n, d)
            y_train: (n,) or (n, 1)

        Returns:
            scalar: log marginal likelihood
        """
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(-1)

        n = X_train.shape[0]
        K = self._compute_Knn(X_train)
        L = torch.linalg.cholesky(K)

        y_centered = y_train - self.mean

        # Solve K \alpha = y_centered via Cholesky: \alpha = K^-1 y
        alpha = torch.cholesky_solve(y_centered, L)

        # Compute quadratic form (y - \mu)^T K^-1 (y - \mu)
        quad = (y_centered.t() @ alpha).squeeze()

        # Compute log determinant: log|K| = 2 * sum(log(diag(L)))
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))

        # Final expression of log marginal likelihood
        log_likelihood = -0.5 * (quad + logdet + n * math.log(2 * math.pi))
        return log_likelihood
