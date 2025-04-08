import torch
import math

class SVGP(torch.nn.Module):
    """
    Stochastic Variational Gaussian Process (SVGP) implementation
    based on Hensman et al. (2013) for scalable approximate inference.

    This model defines a sparse approximation to a full GP using:
        - A set of m inducing points Z in latent space (learnable or fixed)
        - A variational distribution q(f_m) = N(m, S), with S = L @ L^T
        - A kernel function (here RBF) to compute covariance matrices

    Parameters (learnable):
        - log_length_scale : log of RBF length scale l
        - log_signal_var   : log of signal variance \sigma^2
        - log_noise_var    : log of noise variance \epsilon^2
        - Z                : inducing points (m x d)
        - m                : variational mean (m,)
        - raw_L            : lower-triangular raw matrix for covariance S

    ELBO (Evidence Lower Bound) is optimized stochastically using mini-batches:
        ELBO = E_q [log p(y | f)] - KL[q(f_m) || p(f_m)]
    """

    def __init__(self,
                 Z_init_inducing,
                 learn_inducing=True,
                 init_length_scale=1.0,
                 init_var=1.0,
                 init_noise=1e-1):
        super().__init__()

        self.log_length_scale = torch.nn.Parameter(torch.log(torch.tensor(init_length_scale)))
        self.log_signal_var = torch.nn.Parameter(torch.log(torch.tensor(init_var)))
        self.log_noise_var = torch.nn.Parameter(torch.log(torch.tensor(init_noise)))

        self.Z = torch.nn.Parameter(Z_init_inducing.clone().float(), requires_grad=learn_inducing)

        m = Z_init_inducing.shape[0]
        self.m = torch.nn.Parameter(torch.zeros(m))
        self.raw_L = torch.nn.Parameter(torch.eye(m))

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
        X1_sq = (X1**2).sum(dim=1, keepdim=True)
        X2_sq = (X2**2).sum(dim=1, keepdim=True)
        dist_sq = X1_sq - 2 * X1 @ X2.t() + X2_sq.t()
        return self.signal_var * torch.exp(-0.5 * dist_sq / (self.length_scale**2))

    def _compute_Kmm(self):
        """
        Computes the kernel matrix between inducing points (K_mm),
        its stabilized version with jitter, its Cholesky decomposition,
        and its inverse using Cholesky inversion.

        Returns:
            K_mm_jit: Tensor of shape (m, m), stabilized K_mm
            L_mm: Tensor of shape (m, m), lower-triangular Cholesky factor of K_mm_jit
            K_mm_inv: Tensor of shape (m, m), inverse of K_mm_jit
        """
        K_mm = self.rbf_kernel(self.Z, self.Z)
        jitter = 1e-5
        K_mm_jit = K_mm + jitter * torch.eye(K_mm.shape[0], device=K_mm.device)
        L_mm = torch.linalg.cholesky(K_mm_jit)
        K_mm_inv = torch.cholesky_inverse(L_mm)
        return K_mm_jit, L_mm, K_mm_inv

    def _compute_S(self):
        """
        Constructs the variational covariance matrix S from raw_L.

        Returns:
            S: Tensor of shape (m, m), computed as L @ L^T where L = tril(raw_L)
        """
        L = torch.tril(self.raw_L)
        return L @ L.t()

    def kl_term(self):
        """
        Computes the KL divergence between q(f_m) = N(m, S) and the prior p(f_m) = N(0, K_mm).

        Returns:
            KL: Scalar tensor, KL[q(f_m) || p(f_m)]
        """
        K_mm_jit, L_mm, K_mm_inv = self._compute_Kmm()
        S = self._compute_S()
        m_vec = self.m
        M = m_vec.shape[0]

        logdet_K = 2.0 * torch.sum(torch.log(torch.diag(L_mm)))
        L_var = torch.tril(self.raw_L)
        logdet_S = 2.0 * torch.sum(torch.log(torch.diag(L_var)))

        trace_invKS = torch.trace(K_mm_inv @ S)
        m_col = m_vec.unsqueeze(-1)
        mKinv_m = (m_col.t() @ (K_mm_inv @ m_col)).item()

        KL = 0.5 * (trace_invKS + mKinv_m - M + (logdet_K - logdet_S))
        return KL

    def local_lik_terms(self, X_batch, y_batch):
        """
        Computes the expected log likelihood for a mini-batch of observations.

        Args:
            X_batch: Tensor of shape (n, d), batch of inputs
            y_batch: Tensor of shape (n,), batch of targets

        Returns:
            sum_batch: Scalar tensor, total expected log likelihood over the batch
        """
        _, _, K_mm_inv = self._compute_Kmm()
        S = self._compute_S()
        m_col = self.m.unsqueeze(-1)

        K_nM = self.rbf_kernel(X_batch, self.Z)
        alpha_col = K_nM @ (K_mm_inv @ m_col)
        alpha = alpha_col.squeeze(-1)

        sigma2 = self.noise_var
        resid = y_batch - alpha
        ll = -0.5 * ((resid**2)/sigma2 + math.log(2*math.pi*sigma2))

        v = K_nM @ K_mm_inv
        tilde_k = (v * K_nM).sum(dim=1)
        corr1 = -0.5 * sigma2 * tilde_k

        w = v @ S
        trace_sLambda = sigma2 * (v * w).sum(dim=1)
        corr2 = -0.5 * trace_sLambda

        sum_batch = ll + corr1 + corr2
        return sum_batch.sum()

    def elbo(self, X_batch, y_batch, N_total):
        """
        Computes the negative Evidence Lower Bound (ELBO) on the marginal likelihood
        for a mini-batch and scales it to full dataset size.

        Args:
            X_batch: Tensor of shape (n, d), input mini-batch
            y_batch: Tensor of shape (n,), corresponding targets
            N_total: Integer, total number of training examples

        Returns:
            neg_elbo: Scalar tensor, negative ELBO for optimization
        """
        batch_size = X_batch.shape[0]
        sum_loc = self.local_lik_terms(X_batch, y_batch)
        scale = float(N_total) / float(batch_size)
        scaled_loc = scale * sum_loc
        kl_val = self.kl_term()
        elbo_val = scaled_loc - kl_val
        return -elbo_val

    def forward(self, X_train, y_train, X_test):
        """
        Computes predictive mean and marginal variance at test inputs using the variational posterior.

        Args:
            X_train: Tensor of shape (n, d), training inputs
            y_train: Tensor of shape (n,), training targets
            X_test: Tensor of shape (n_test, d), test inputs

        Returns:
            mean: Tensor of shape (n_test,), predictive mean
            var_diag: Tensor of shape (n_test,), predictive variance
        """
        K_mm_jit, L_mm, K_mm_inv = self._compute_Kmm()
        S = self._compute_S()

        K_xM = self.rbf_kernel(X_test, self.Z)
        m_col = self.m.unsqueeze(-1)
        mean_col = K_xM @ (K_mm_inv @ m_col)
        mean = mean_col.squeeze(-1)

        N_test = X_test.shape[0]
        var_diag = self.signal_var * torch.ones(N_test, device=X_test.device)

        A = K_xM @ K_mm_inv
        sub = (A * K_xM).sum(dim=1)

        AS = A @ S
        add = (AS * A).sum(dim=1)

        var_diag = var_diag - sub + add
        return mean, var_diag
