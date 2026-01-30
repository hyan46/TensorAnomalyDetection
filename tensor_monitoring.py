"""
Tensor-based process monitoring via low-rank decomposition (Yan et al., IEEE TASE 2015).

Supports tensors of any order: X shape (N, d1, d2, ..., d_m) with sample mode first.

- UPCA: unfold each sample to vector, PCA; monitor T² and Q.
- MPCA: Tucker on (d1, ..., d_m, N); projection factors per mode; monitor core features and Q.
- TROD: CP on (d1, ..., d_m, N); spatial rank-one factors; monitor sample weights and Q.

Control limits: empirical (1-alpha) percentiles; alpha≈0.005 for in-control ARL ≈ 200.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class BaseTensorControlChart(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mean_ = None
        self.cov_ = None
        self.inv_cov_ = None
        self.ucl_t2_ = None
        self.ucl_q_ = None

    def _compute_t2_q_limits(self, features, residuals):
        n = features.shape[0]
        p = features.shape[1]
        
        # Calculate mean and covariance
        self.mean_ = np.mean(features, axis=0)
        self.cov_ = np.cov(features, rowvar=False)
        # Add small regularization to covariance if singular
        if np.linalg.cond(self.cov_) > 1e10:
             self.cov_ += np.eye(p) * 1e-6
        self.inv_cov_ = np.linalg.pinv(self.cov_)
        
        # Compute T2 stats for training data
        t2_stats = []
        for i in range(n):
            diff = features[i] - self.mean_
            t2 = diff.T @ self.inv_cov_ @ diff
            t2_stats.append(t2)
        t2_stats = np.array(t2_stats)
        
        # Paper: "empirical distributions ... (1-alpha) percentiles ... are defined as control limits"
        # Combined in-control ARL ≈ 200 => alpha ≈ 0.005 per chart
        self.ucl_t2_ = np.percentile(t2_stats, 100 * (1 - self.alpha))

        q_stats = residuals
        self.ucl_q_ = np.percentile(q_stats, 100 * (1 - self.alpha))
            
        return t2_stats, q_stats

    def predict(self, X):
        features, residuals = self.transform(X)
        
        t2_stats = []
        for i in range(len(features)):
            diff = features[i] - self.mean_
            t2 = diff.T @ self.inv_cov_ @ diff
            t2_stats.append(t2)
        t2_stats = np.array(t2_stats)
        
        q_stats = residuals
        
        t2_alarms = t2_stats > self.ucl_t2_
        q_alarms = q_stats > self.ucl_q_
        
        return t2_alarms, q_alarms, t2_stats, q_stats

class UPCAControlChart(BaseTensorControlChart):
    """Unfolded PCA: flatten each sample to a vector, then PCA. Works for any tensor order (N, d1, ..., d_m)."""

    def __init__(self, n_components=None, alpha=0.01):
        super().__init__(alpha=alpha)
        self.n_components = n_components
        self.pca = None
        self.shape_ = None

    def fit(self, X, y=None):
        # X: (n_samples, d1, d2, ..., d_m) — any order
        X = np.array(X)
        self.shape_ = X.shape[1:]
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        
        self.pca = PCA(n_components=self.n_components)
        features = self.pca.fit_transform(X_flat)
        
        # Reconstruct to compute residuals
        X_recon = self.pca.inverse_transform(features)
        residuals_matrix = X_flat - X_recon
        spe = np.sum(residuals_matrix**2, axis=1)
        
        self._compute_t2_q_limits(features, spe)
        return self

    def transform(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        
        features = self.pca.transform(X_flat)
        X_recon = self.pca.inverse_transform(features)
        residuals_matrix = X_flat - X_recon
        spe = np.sum(residuals_matrix**2, axis=1)
        
        return features, spe

class MPCAControlChart(BaseTensorControlChart):
    """MPCA: Tucker on (d1, ..., d_m, N); one projection matrix per mode. Any order (N, d1, ..., d_m)."""

    def __init__(self, rank, alpha=0.01, centered=False):
        super().__init__(alpha=alpha)
        self.rank = rank  # tuple (R1, ..., R_m), length = number of non-sample modes
        self.centered = centered
        self.mean_tensor_ = None
        self.factors_ = None  # list of projection matrices [U1, ..., U_m]
        self.n_modes_ = None

    def fit(self, X, y=None):
        X = np.array(X)
        if X.ndim < 2:
            raise ValueError("X must have shape (n_samples, d1, ..., d_m) with at least 2 dims.")
        N = X.shape[0]
        self.n_modes_ = X.ndim - 1
        if len(self.rank) != self.n_modes_:
            raise ValueError(f"rank must have length {self.n_modes_} for X.shape {X.shape}.")

        self.mean_tensor_ = np.mean(X, axis=0)
        if self.centered:
            X_process = X - self.mean_tensor_
        else:
            X_process = X

        # Stack as (d1, ..., d_m, N) — sample mode last (paper convention).
        perm = tuple(range(1, X.ndim)) + (0,)
        T = np.transpose(X_process, perm)

        tucker_rank = tuple(self.rank) + (N,)
        core, factors = tucker(T, rank=tucker_rank)

        # factors[0]...(factors[n_modes-1]) are spatial; factors[n_modes] is sample mode.
        self.factors_ = list(factors[: self.n_modes_])

        # Features: core has shape (R1, ..., R_m, N); slice per sample.
        features = np.array([np.asarray(core[..., i]).flatten() for i in range(N)])

        residuals = []
        for i in range(N):
            core_i = np.asarray(core[..., i])
            rec = tl.tenalg.multi_mode_dot(core_i, self.factors_)
            res = np.asarray(T[..., i]) - rec
            residuals.append(float(np.sum(res ** 2)))
        residuals = np.array(residuals)

        self._compute_t2_q_limits(features, residuals)
        return self

    def transform(self, X):
        X = np.array(X)
        if self.centered:
            X_process = X - self.mean_tensor_
        else:
            X_process = X

        projection_factors = [f.T for f in self.factors_]
        features = []
        residuals = []
        for i in range(X.shape[0]):
            tensor = X_process[i]
            core_i = tl.tenalg.multi_mode_dot(tensor, projection_factors)
            features.append(np.asarray(core_i).flatten())
            rec = tl.tenalg.multi_mode_dot(core_i, self.factors_)
            res = tensor - rec
            residuals.append(float(np.sum(res ** 2)))
        return np.array(features), np.array(residuals)

class TRODControlChart(BaseTensorControlChart):
    """TROD: CP on (d1, ..., d_m, N); shared spatial rank-one factors; features = sample weights. Any order."""

    def __init__(self, rank, alpha=0.01):
        super().__init__(alpha=alpha)
        self.rank = rank  # number of rank-1 components (int)
        self.spatial_factors_ = None  # list of (d_k, R) matrices
        self.weights_ = None
        self.H_gram_inv_ = None  # (H.T @ H)^{-1} for projection

    def fit(self, X, y=None):
        X = np.array(X)
        if X.ndim < 2:
            raise ValueError("X must have shape (n_samples, d1, ..., d_m) with at least 2 dims.")
        N = X.shape[0]
        n_modes = X.ndim - 1

        # Stack as (d1, ..., d_m, N); sample mode last.
        perm = tuple(range(1, X.ndim)) + (0,)
        T = np.transpose(X, perm)

        weights, factors = parafac(T, rank=self.rank, init="random")
        self.weights_ = np.array(weights)

        # factors[0]...(n_modes-1) = spatial; factors[n_modes] = sample (N, R).
        self.spatial_factors_ = list(factors[:n_modes])
        sample_factor = factors[n_modes]
        features = sample_factor * self.weights_  # (N, R)

        H = tl.tenalg.khatri_rao(self.spatial_factors_, skip_matrix=None)
        H = np.asarray(H)
        self.H_gram_inv_ = np.linalg.pinv(H.T @ H)

        residuals = []
        for i in range(N):
            lambdas = features[i]
            rec_i = tl.cp_to_tensor((lambdas, self.spatial_factors_))
            res = np.asarray(T[..., i]) - rec_i
            residuals.append(float(np.sum(res ** 2)))
        residuals = np.array(residuals)

        self._compute_t2_q_limits(features, residuals)
        return self

    def transform(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        R = self.rank

        features = []
        residuals = []
        for i in range(n_samples):
            Xi = X[i]
            rhs = np.zeros(R)
            for r in range(R):
                val = np.asarray(Xi).copy()
                for mode, U in enumerate(self.spatial_factors_):
                    val = tl.tenalg.mode_dot(val, np.asarray(U[:, r]), mode=0)
                rhs[r] = float(val)
            lambdas = self.H_gram_inv_ @ rhs
            features.append(lambdas)

            rec_i = tl.cp_to_tensor((lambdas, self.spatial_factors_))
            res = np.asarray(Xi) - rec_i
            residuals.append(float(np.sum(res ** 2)))

        return np.array(features), np.array(residuals)

