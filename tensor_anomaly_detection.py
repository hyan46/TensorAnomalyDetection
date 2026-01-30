import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker
from tensorly.tenalg.proximal import soft_thresholding
from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class TensorAnomalyDetection(BaseEstimator, OutlierMixin, TransformerMixin):
    def __init__(self, rank=None, lambda_param=0.1, max_iter=100, tol=1e-6):
        """
        Initialize the TensorAnomalyDetection class.

        Parameters:
        - rank: List[int], rank for each mode in Tucker decomposition. If None, defaults to half of each mode size.
        - lambda_param: float, regularization parameter for anomaly component.
        - max_iter: int, maximum number of iterations for optimization.
        - tol: float, convergence tolerance.
        """
        self.rank = rank
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """
        Fit the model to the training tensor data X.

        Parameters:
        - X: array-like, shape (n_samples, ...) tensor data.
        - y: Ignored.

        Returns:
        - self: object
        """
        # Validate input
        X = self._validate_inputs(X)
        self.X_shape_ = X.shape[1:]  # Assuming X shape is (n_samples, ...)

        # Initialize A as zeros
        A = np.zeros(self.X_shape_)

        # Initialize rank if not provided
        if self.rank is None:
            self.rank_ = [dim // 2 for dim in self.X_shape_]
        else:
            self.rank_ = self.rank

        # Perform three-way decomposition
        C, U, A_recovered = self._three_way_decomposition(
            X, A, self.rank_, self.lambda_param, self.max_iter, self.tol
        )

        self.C_ = C
        self.U_ = U
        self.A_recovered_ = A_recovered

        # Compute threshold for anomaly detection
        self.threshold_ = self._compute_threshold(A_recovered)

        return self

    def predict(self, X):
        """
        Predict whether each tensor in X is an outlier or not.

        Parameters:
        - X: array-like, shape (n_samples, ...) tensor data.

        Returns:
        - y_pred: array-like, shape (n_samples,)
          Returns -1 for outliers and 1 for inliers.
        """
        check_is_fitted(self, ['C_', 'U_', 'A_recovered_', 'threshold_'])
        X = self._validate_inputs(X)

        y_pred = []
        for tensor in X:
            # Reconstruct background
            X_reconstructed = tl.tucker_to_tensor((self.C_, self.U_))
            # Compute anomaly tensor
            A = soft_thresholding(tensor - X_reconstructed, self.lambda_param)
            # Threshold to determine outlier
            anomaly_score = tl.norm(A)
            label = -1 if anomaly_score > self.threshold_ else 1
            y_pred.append(label)
        
        return np.array(y_pred)

    def _three_way_decomposition(self, X, A, rank, lambda_param, max_iter, tol):
        """
        Perform three-way decomposition on the tensor data.

        Parameters:
        - X: array-like, shape (n_samples, ...) tensor data.
        - A: Initial anomaly tensor.
        - rank: List[int], rank for Tucker decomposition.
        - lambda_param: float, regularization parameter.
        - max_iter: int, maximum iterations.
        - tol: float, convergence tolerance.

        Returns:
        - C: Core tensor from Tucker decomposition.
        - U: List of factor matrices.
        - A: Recovered anomaly tensor.
        """
        # Initialize A as mean of X
        A = np.zeros(self.X_shape_)
        
        # Initialize U by performing Tucker decomposition on the mean tensor
        mean_tensor = np.mean(X, axis=0)
        
        # Add very small noise to avoid numerical issues
        mean_tensor += np.random.normal(0, 1e-10, mean_tensor.shape)
        
        # Initial decomposition
        C, U = non_negative_tucker(mean_tensor, rank=rank, n_iter_max=100, tol=1e-6)
        
        prev_loss = float('inf')
        for iteration in range(max_iter):
            # Step 1: Optimize U given A
            Y_minus_A = mean_tensor - A
            C, U = non_negative_tucker(Y_minus_A, rank=rank, n_iter_max=100, tol=1e-6)
            
            # Step 2: Optimize A given U
            reconstructed = tl.tucker_to_tensor((C, U))
            residual = mean_tensor - reconstructed
            
            # Apply soft thresholding with scaled lambda
            # Scale lambda_param by the magnitude of the residual
            scaled_lambda = lambda_param * np.mean(np.abs(residual))
            A_new = soft_thresholding(residual, scaled_lambda)
            
            # Compute current loss
            current_loss = tl.norm(mean_tensor - reconstructed - A_new)
            
            # Check for convergence
            if abs(current_loss - prev_loss) < tol:
                print(f"Converged after {iteration + 1} iterations.")
                break
                
            prev_loss = current_loss
            A = A_new
        
        return C, U, A

    def _compute_threshold(self, A_recovered):
        """
        Compute the threshold for anomaly detection based on recovered A.

        Parameters:
        - A_recovered: Anomaly tensor.

        Returns:
        - threshold: float, computed threshold.
        """
        # Use a more robust threshold based on the distribution of values
        return np.mean(np.abs(A_recovered)) + 2 * np.std(np.abs(A_recovered))

    def _validate_inputs(self, X):
        """
        Validate the input tensor data.

        Parameters:
        - X: array-like, tensor data.

        Returns:
        - X: np.ndarray, validated tensor data.
        """
        X = np.array(X)
        if X.ndim < 2:
            raise ValueError("Input tensor X must have at least 2 dimensions (n_samples, ...) ")
        if self.rank is not None and len(self.rank) != X.ndim - 1:
            raise ValueError("Length of rank must match the tensor dimensions excluding the sample axis.")
        return X

    def score_samples(self, X):
        """
        Compute the anomaly scores for each sample in X.

        Parameters:
        - X: array-like, shape (n_samples, ...) tensor data.

        Returns:
        - scores: array-like, shape (n_samples,)
          The anomaly scores for each sample.
        """
        check_is_fitted(self, ['C_', 'U_', 'A_recovered_', 'threshold_'])
        X = self._validate_inputs(X)

        scores = []
        for tensor in X:
            X_reconstructed = tl.tucker_to_tensor((self.C_, self.U_))
            A = soft_thresholding(tensor - X_reconstructed, self.lambda_param)
            anomaly_score = tl.norm(A)
            scores.append(anomaly_score)
        
        return np.array(scores)

    def decision_function(self, X):
        """
        Compute the decision function for each sample in X.
        The decision function is the difference between the threshold
        and the anomaly score.

        Parameters:
        - X: array-like, shape (n_samples, ...) tensor data.

        Returns:
        - decision: array-like, shape (n_samples,)
          The decision function for each sample.
          Negative values indicate outliers.
        """
        return self.threshold_ - self.score_samples(X)