import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from tensorly.decomposition import non_negative_tucker
from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from tensorly.random import random_tensor
from sklearn.metrics import f1_score

class TensorHotellingT2(BaseEstimator, OutlierMixin, TransformerMixin):
    def __init__(self, rank, alpha=0.05, n_iter_max=100, tol=1e-6):
        """
        Initialize the TensorHotellingT2 class.

        Parameters:
        - rank: list or int, the rank(s) for the Tucker decomposition.
        - alpha: float, significance level for the empirical threshold (e.g., 95% quantile).
        - n_iter_max: int, maximum iterations for decomposition.
        - tol: float, convergence tolerance for decomposition.
        """
        self.rank = rank
        self.alpha = alpha
        self.n_iter_max = n_iter_max
        self.tol = tol

    def fit(self, X, y=None):
        """
        Fit the model to the training tensor data X.

        Parameters:
        - X: list or array of tensors, shape (n_samples, tensor_shape)
        """
        # Validate input
        X = self._validate_tensors(X)

        # Decompose tensors and collect core tensors
        core_tensors = []
        self.reconstructed_tensors_ = []
        for tensor in X:
            core, factors = non_negative_tucker(
                tensor, rank=self.rank, n_iter_max=self.n_iter_max, tol=self.tol
            )
            core_vector = tl.tensor_to_vec(core)
            core_tensors.append(core_vector)
            # Reconstruct the tensor
            reconstructed = tl.tucker_to_tensor((core, factors))
            self.reconstructed_tensors_.append(reconstructed)
        self.core_tensors_ = np.array(core_tensors)
        self.reconstructed_tensors_ = np.array(self.reconstructed_tensors_)

        # Compute mean and covariance of core tensors
        self.mean_ = np.mean(self.core_tensors_, axis=0)
        self.cov_ = np.cov(self.core_tensors_, rowvar=False)

        # Handle singular covariance matrix
        self.inverse_cov_ = np.linalg.pinv(self.cov_)

        # Compute residuals for training data
        self.residuals_ = self._compute_residuals(X, self.reconstructed_tensors_)

        return self

    def score_samples(self, X):
        """
        Compute the T² statistic for each tensor in X.

        Parameters:
        - X: list or array of tensors, shape (n_samples, tensor_shape)

        Returns:
        - t2_scores: array of T² statistics, shape (n_samples,)
        """
        check_is_fitted(self, ['mean_', 'inverse_cov_'])
        X = self._validate_tensors(X)

        t2_scores = []
        for tensor in X:
            core, _ = non_negative_tucker(
                tensor, rank=self.rank, n_iter_max=self.n_iter_max, tol=self.tol
            )
            core_vector = tl.tensor_to_vec(core)
            diff = core_vector - self.mean_
            t2 = diff.T @ self.inverse_cov_ @ diff
            t2_scores.append(t2)
        return np.array(t2_scores)

    def reconstruction_error(self, X):
        """
        Compute the residual sum of squares (RSS) for each tensor in X.

        Parameters:
        - X: list or array of tensors, shape (n_samples, tensor_shape)

        Returns:
        - rss: array of RSS values, shape (n_samples,)
        """
        X = self._validate_tensors(X)

        residuals = []
        for tensor in X:
            core, factors = non_negative_tucker(
                tensor, rank=self.rank, n_iter_max=self.n_iter_max, tol=self.tol
            )
            reconstructed = tl.tucker_to_tensor((core, factors))
            # Compute RSS
            rss = tl.norm(tensor - reconstructed, 2)**2
            residuals.append(rss)
        return np.array(residuals)

    def _compute_ucl(self, method='T2'):
        """
        Compute the Upper Control Limit (UCL) based on the empirical data-driven
        threshold. The UCL is set as the (1 - alpha) quantile of the corresponding
        statistic from the training data.

        Parameters:
        - method: str, 'T2' for using the T² statistic or 'residual' for using RSS.

        Returns:
        - ucl: float, the computed empirical threshold.
        """
        if method == 'T2':
            # Use the 95th percentile of the training T² statistics as the threshold
            t2_scores = self.score_samples(self.reconstructed_tensors_)
            ucl = np.percentile(t2_scores, 100 * (1 - self.alpha))
        elif method == 'residual':
            # Use the 95th percentile of the training residuals as the threshold
            ucl = np.percentile(self.residuals_, 100 * (1 - self.alpha))
        else:
            raise ValueError("Method must be either 'T2' or 'residual'")
        
        return ucl

    def predict(self, X, method='T2'):
        """
        Predict whether each tensor in X is an outlier or not based on the specified method.

        Parameters:
        - X: list or array of tensors, shape (n_samples, tensor_shape)
        - method: str, 'T2' for using the T² statistic or 'residual' for using RSS.

        Returns:
        - y_pred: array-like, shape (n_samples,)
          Returns -1 for outliers and 1 for inliers.
        """

        if method == 'T2':
            # Compute T² statistics for each sample
            ucl = self._compute_ucl(method)
            scores = self.score_samples(X)
            return np.where(scores > ucl, -1, 1)
        elif method == 'residual':
            # Compute residuals for each sample
            ucl = self._compute_ucl(method)
            scores = self.reconstruction_error(X)
            return np.where(scores > ucl, -1, 1)
        elif method == 'both':
            ucl_t2 = self._compute_ucl(method='T2')
            ucl_residual = self._compute_ucl(method='residual')
            scores_t2 = self.score_samples(X)
            scores_residual = self.reconstruction_error(X)
            # Use numpy's logical_or to compare arrays element-wise
            outlier_mask = np.logical_or(scores_t2 > ucl_t2, scores_residual > ucl_residual)
            return np.where(outlier_mask, -1, 1)
        else:
            raise ValueError("Method must be 'T2', 'residual', or 'both'")

        # Flag as -1 for outliers and 1 for inliers based on the empirical threshold

    def _compute_residuals(self, X, reconstructed_tensors):
        """
        Compute the residuals (RSS) between original and reconstructed tensors.
        """
        residuals = []
        for original, reconstructed in zip(X, reconstructed_tensors):
            rss = tl.norm(original - reconstructed, 2)**2
            residuals.append(rss)
        return np.array(residuals)

    def _validate_tensors(self, X):
        """
        Validate input tensors.
        """
        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("Input must be a list or array of tensors.")
        return X


if __name__ == "__main__":
    # Parameters
    n_samples = 100
    tensor_shape = (10, 10, 10)
    rank = [2, 2, 2]  # Tucker rank for decomposition

    # Generate the initial synthetic tensor data
    X = [random_tensor(tensor_shape, rank=rank, full=True) for _ in range(n_samples)]

    # Add a mean shift of 10 to each element in the last 50 samples to create outliers
    X_outliers = [tensor + 10 for tensor in X[-50:]]

    # Split into normal training samples and combined test samples (normal + outliers)
    X_train = X[:n_samples - 50]
    X_combined = X_train + X_outliers

    # Verifying the mean shift on the last few samples
    print("Mean of last few normal samples:")
    for tensor in X_combined[:5]:
        print(tl.mean(tensor).item())

    print("\nMean of last few shifted samples:")
    for tensor in X_combined[-5:]:
        print(tl.mean(tensor).item())

    # Fit the model on normal training data
    model = TensorHotellingT2(rank=rank, alpha=0.05)
    model.fit(X_train)

    # Ground truth labels: 1 for normal samples, -1 for outliers
    true_labels = np.array([1] * (n_samples - 50) + [-1] * 50)

    # Predict outliers in test data using T² statistics
    predictions_t2 = model.predict(X_combined, method='T2')
    print("Outliers using T² method:", predictions_t2)

    # Calculate F1-score for T² method
    f1_t2 = f1_score(true_labels, predictions_t2, pos_label=-1)
    print(f"F1-score using T² method: {f1_t2:.2f}")

    # Predict outliers in test data using Residual (RSS) statistics
    predictions_residual = model.predict(X_combined, method='residual')
    print("Outliers using Residual method:", predictions_residual)

    # Predict outliers in test data using both T² and Residual (RSS) statistics
    predictions_both = model.predict(X_combined, method='both')
    print("Outliers using both methods:", predictions_both)


    # Calculate F1-score for Residual method
    f1_residual = f1_score(true_labels, predictions_residual, pos_label=-1)
    print(f"F1-score using Residual method: {f1_residual:.2f}")

    # Calculate F1-score for both methods
    f1_both = f1_score(true_labels, predictions_both, pos_label=-1)
    print(f"F1-score using both methods: {f1_both:.2f}")

