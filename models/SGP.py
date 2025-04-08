import torch
import math

class SGP(torch.nn.Module):
    """
    Sparse Gaussian Process (approx. Titsias, 2009).

    Prediction equations:
        mean(x*) = k(x*, Z) K_mm^{-1} mu
        var(x*)  = k(x*, x*) - k(x*, Z) K_mm^{-1} k(Z, x*) + k(x*, Z) K_mm^{-1} S K_mm^{-1} k(Z, x*).

    Variational lower bound (ELBO):
        ELBO = log N(y | 0, Q_nn + sigma^2 I) - (1/(2 sigma^2)) trace(K_nn - Q_nn),
    with Q_nn = K_nm K_mm^{-1} K_mn.

    Hyperparameters:
      - log_length_scale
      - log_signal_var
      - log_noise_var

    Inducing points (Z): shape (m, d)
    """

    def __init__(self,
                 Z_init_inducing,
                 init_length_scale=1.0,
                 init_var=1.0,
                 init_noise=1e-2,
                 learn_inducing=True):
        super().__init__()
        
        self.log_length_scale = torch.nn.Parameter(
            torch.log(torch.tensor(init_length_scale, dtype=torch.float32))
        )
        self.log_signal_var = torch.nn.Parameter(
            torch.log(torch.tensor(init_var, dtype=torch.float32))
        )
        self.log_noise_var = torch.nn.Parameter(
            torch.log(torch.tensor(init_noise, dtype=torch.float32))
        )
        
        self.Z = torch.nn.Parameter(
            Z_init_inducing.clone().float(),
            requires_grad=learn_inducing
        )

    @property
    def length_scale(self):
        return torch.exp(self.log_length_scale)

    @property
    def signal_var(self):
        return torch.exp(self.log_signal_var)

    @property
    def noise_var(self):
        return torch.exp(self.log_noise_var)
    
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
        
        return self.signal_var * torch.exp(-0.5 * dist_sq / (self.length_scale**2))
    
    def _compute_Kmm(self):
        """
        Computes the prior covariance matrix K_mm between inducing points (self.Z),
        adds a small jitter for numerical stability, and returns:
            - the kernel matrix K_mm,
            - its inverse via Cholesky decomposition,
            - and the Cholesky factor L_mm (such that K_mm = L_mm @ L_mm^T).

        Returns:
            K_mm       : (m, m) covariance matrix
            K_mm_inv   : (m, m) inverse of K_mm via Cholesky
            L_mm       : (m, m) lower triangular Cholesky factor of K_mm
        """
        K_mm = self.rbf_kernel(self.Z, self.Z)
        jitter = 1e-5
        K_mm_jittered = K_mm + jitter * torch.eye(K_mm.shape[0], device=K_mm.device)
        
        L_mm = torch.linalg.cholesky(K_mm_jittered)
        K_mm_inv = torch.cholesky_inverse(L_mm)
        
        return K_mm, K_mm_inv
    
    def _compute_Q_nn(self, X, K_mm_inv):
        """
        Computes the low-rank Nyström approximation Q_nn of the full covariance matrix K_nn.

        Q_nn = K_nm @ K_mm^{-1} @ K_mn

        where:
            - K_nm = kernel matrix between inputs X and inducing points Z
            - K_mm_inv is the inverse of the prior covariance between inducing points

        Args:
            X         : (n, d) input data
            K_mm_inv : (m, m) inverse of the kernel matrix between inducing points

        Returns:
            Q_nn : (n, n) low-rank approximation of the full covariance matrix
        """
        K_nm = self.rbf_kernel(X, self.Z)     # (n, m)
        K_mn = K_nm.t()                       # (m, n)

        Q_nn = K_nm @ (K_mm_inv @ K_mn)       # (n, n)
        return Q_nn
    
    def _variational_dist_fm(self, X):
        """
        Internal helper function to compute the posterior covariance S
        of the optimal variational distribution q(f_m) = N(mu, S),
        where f_m are the latent function values at the inducing points.

        The optimal closed-form expression for the covariance is:
            S = K_mm @ (I + (1/\sigma^2) * K_mn @ K_nm)^(-1) @ K_mm

        This corresponds to the analytical solution derived in Titsias (2009),
        which minimizes the Kullback-Leibler divergence to the true posterior.

        Args:
            X : (n, d) training inputs used to compute K_nm and K_mn

        Returns:
            K_mm     : (m, m) prior covariance between inducing points
            K_mm_inv : (m, m) inverse of K_mm
            K_nm     : (n, m) covariance between training points and inducing points
            S        : (m, m) optimal posterior covariance of q(f_m)
        """
        # Compute prior covariance and its inverse between inducing points
        K_mm, K_mm_inv = self._compute_Kmm()

        # Compute cross-covariances between data and inducing points
        K_nm = self.rbf_kernel(X, self.Z)  # shape (n, m)
        K_mn = K_nm.t()                    # shape (m, n)

        sigma_sq = self.noise_var
        m = K_mm.shape[0]
        I_m = torch.eye(m, device=K_mm.device)

        # Compute intermediate matrix: Sigma_m = I + (1/\sigma^2) K_mn @ K_nm
        Sigma_m = I_m + (1.0 / sigma_sq) * (K_mn @ K_nm)

        # Invert via Cholesky for stability: (Sigma_m)^-1
        L_Sigma_m = torch.linalg.cholesky(Sigma_m)
        Sigma_m_inv = torch.cholesky_inverse(L_Sigma_m)

        # Final posterior covariance: S = K_mm @ Sigma_m^-1 @ K_mm
        S = K_mm @ Sigma_m_inv @ K_mm

        return K_mm, K_mm_inv, K_nm, S
    
    def variational_dist_fm(self, X, y):
        """
        Computes the optimal variational distribution q(f_m) = N(mu, S)
        given training data (X, y), using closed-form expressions.

        The optimal parameters are:
            S  = K_mm @ (I + (1/\sigma^2) * K_mn @ K_nm)^(-1) @ K_mm
            mu = (1/\sigma^2) * S @ K_mm⁻¹ @ K_mn @ y

        Args:
            X : (n, d) training inputs
            y : (n,) or (n, 1) training outputs

        Returns:
            mu : (m, 1) variational mean of f_m
            S  : (m, m) variational covariance of f_m
        """
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # Ensure y is a column vector

        # Compute prior + optimal posterior covariance S
        K_mm, K_mm_inv, K_nm, S = self._variational_dist_fm(X)
        sigma_sq = self.noise_var

        # Compute mu = (1/\sigma^2) * S @ K_mm^-1 @ K_mn @ y
        A = K_mm_inv @ K_nm.t()   # shape: (m, n)
        Ay = A @ y                # shape: (m, 1)
        mu = (1.0 / sigma_sq) * (S @ Ay)

        return mu, S

    
    def elbo(self, X, y):
        """
        Computes the Evidence Lower Bound (ELBO) for the Sparse Gaussian Process.

        ELBO = log N(y | 0, Q_nn + sigma^2 I) - (1 / (2 sigma^2)) * Tr(K_nn - Q_nn)
        where:
            - Q_nn = Knm Kmm^{-1} Kmn is the Nyström low-rank approximation of K_nn
            - sigma^2 is the observation noise variance
            - K_nn is the full RBF covariance matrix between inputs
            - The trace term penalizes the difference between the true and approximated covariance

        The log-likelihood term is given by:
            log N(y | 0, S) = -1/2 y^T S^{-1} y - 1/2 log det(S) - n/2 log(2 \pi)
            where S = Q_nn + sigma^2 * I
        """
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # Ensure y is a column vector of shape (n, 1)
        n = X.shape[0]
        
        # Compute the inverse of the prior covariance between inducing points
        _, K_mm_inv = self._compute_Kmm()
        
        # Compute Q_nn = K_nm K_mm^{-1} K_mn
        Q_nn = self._compute_Q_nn(X, K_mm_inv)
        
        # Define the approximate covariance of the marginal likelihood: S = Q_nn + sigma^2 * I
        jitter = 1e-5
        Sigma = Q_nn + (self.noise_var + jitter) * torch.eye(n, device=X.device)

        # Perform Cholesky decomposition: Sigma = L L^T
        L_Sigma = torch.linalg.cholesky(Sigma)

        # Solve for alpha = Sigma^{-1} y using the Cholesky factor
        alpha = torch.cholesky_solve(y, L_Sigma)
        
        # Compute the quadratic form y^T Sigma^{-1} y
        y_SigmaInv_y = (y.t() @ alpha).squeeze()  # Result is a scalar, keeps gradient path

        # Compute log determinant of Sigma from Cholesky factor: log|S| = 2 * sum(log(diag(L)))
        logdet_Sigma = 2.0 * torch.sum(torch.log(torch.diag(L_Sigma)))
        
        # Log-marginal likelihood term: log N(y | 0, S)
        term_logN = -0.5 * (y_SigmaInv_y + logdet_Sigma + n * math.log(2 * math.pi))
        
        # Compute the full covariance matrix
        K_nn = self.rbf_kernel(X, X)

        # Compute trace(K_nn - Q_nn)
        trace_diff = torch.trace(K_nn - Q_nn)
        
        # Variational correction term: - (1/(2 sigma^2)) * Tr(K_nn - Q_nn)
        term_trace = -0.5 * (1.0 / self.noise_var) * trace_diff
        
        # Final ELBO = log-likelihood term - trace penalty
        return term_logN + term_trace
    
    def forward(self, X_train, y_train, X_test):
        """
        Compute predictive mean and variance at test locations X_test using the
        variational posterior over inducing outputs f_m.

        Predictive equations:
            mean(x*) = k_x*Z @ K_mm^-1 @ mu
            var(x*)  = k(x*,x*) - k_x*Z @ K_mm^-1 @ k_Zx*
                    + k_x*Z @ K_mm^-1 @ S @ K_mm^-1 @ k_Zx*

        Args:
            X_train : training inputs used to compute the variational posterior
            y_train : training outputs
            X_test  : test inputs at which predictions are computed

        Returns:
            mean     : predictive mean at each test input (shape: [n_test])
            var_diag : predictive marginal variance at each test input (shape: [n_test])
        """
        # Compute the variational posterior parameters: q(f_m) = N(mu, S)
        mu, S = self.variational_dist_fm(X_train, y_train)
        
        # Compute prior covariance and its inverse at inducing inputs Z
        _, K_mm_inv = self._compute_Kmm()

        # Cross-covariance between test points and inducing inputs
        K_xsZ = self.rbf_kernel(X_test, self.Z)

        # Predictive mean: E[f(x*)] = k_x*Z @ K_mm^-1 @ mu
        mean_col = K_xsZ @ (K_mm_inv @ mu)
        mean = mean_col.squeeze(-1)

        # Compute predictive variance (diagonal only)
        n_test = X_test.shape[0]
        var_diag = self.signal_var * torch.ones(n_test, device=X_test.device)

        # Common factor: k_x*Z @ K_mm^-1 (used in both sub and add)
        left = K_xsZ @ K_mm_inv

        # First correction term: - k_x*Z @ K_mm^-1 @ k_Zx* (standard GP variance reduction)
        sub = (left * K_xsZ).sum(dim=1)

        # Variational uncertainty correction: + k_x*Z @ K_mm^-1 @ S @ K_mm^-1 @ k_Zx*
        vSv = (left @ S) * left
        add = vSv.sum(dim=1)

        # Final predictive variance
        var_diag = var_diag - sub + add + self.noise_var

        return mean, var_diag

