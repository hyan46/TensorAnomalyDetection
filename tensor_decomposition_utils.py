import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg.proximal import soft_thresholding
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def create_low_rank_tensor(shape, rank, random_state=None):
    """
    Create a low-rank tensor using Tucker decomposition.

    Parameters:
    - shape: Tuple indicating the shape of the tensor.
    - rank: Tuple indicating the rank for each mode.
    - random_state: Seed for reproducibility.

    Returns:
    - tensor: Low-rank tensor.
    - core: Core tensor.
    - factors: Factor matrices.
    """
    if random_state is not None:
        np.random.seed(random_state)
    core = np.random.rand(*rank)
    factors = [np.random.rand(shape[i], rank[i]) for i in range(len(shape))]
    tensor = tl.tucker_to_tensor((core, factors))
    return tensor, core, factors

def create_sparse_anomaly(shape, sparsity=0.05, random_state=None):
    """
    Create a sparse anomaly tensor.

    Parameters:
    - shape: Tuple indicating the shape of the tensor.
    - sparsity: Fraction of elements to be non-zero.
    - random_state: Seed for reproducibility.

    Returns:
    - A: Sparse anomaly tensor.
    - A_binary: Binary tensor indicating anomaly locations.
    """
    if random_state is not None:
        np.random.seed(random_state)
    A = np.zeros(shape)
    num_elements = np.prod(shape)
    num_anomalies = int(sparsity * num_elements)
    indices = np.unravel_index(
        np.random.choice(num_elements, num_anomalies, replace=False),
        shape
    )
    A[indices] = np.random.randn(num_anomalies)  # Anomalies with random magnitudes
    A_binary = (A != 0).astype(int)
    return A, A_binary

def evaluate_f1_score_true_pred(true_anomalies, predicted_anomalies):
    """
    Evaluate the F1 score for anomaly detection.

    Parameters:
    - true_anomalies: Binary tensor indicating true anomalies.
    - predicted_anomalies: Binary tensor indicating predicted anomalies.

    Returns:
    - f1: F1 score.
    """
    true_flat = true_anomalies.flatten()
    predicted_flat = predicted_anomalies.flatten()
    return f1_score(true_flat, predicted_flat)

def visualize_tensors(X, A_true, Y, model, A_recovered, A_pred):
    """
    Visualize slices of the tensors.
    
    Parameters:
    - X: Ground truth background tensor
    - A_true: Ground truth anomaly tensor
    - Y: Observed tensor
    - model: Fitted TensorAnomalyDetection model
    - A_recovered: Recovered anomaly tensor
    - A_pred: Predicted anomaly tensor (binary)
    """
    # Take middle slice for visualization
    slice_idx = X.shape[-1] // 2

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # First row
    axs[0, 0].imshow(X[:, :, slice_idx], cmap='gray')
    axs[0, 0].set_title('Ground Truth Background (X)')

    axs[0, 1].imshow(A_true[:, :, slice_idx], cmap='hot')
    axs[0, 1].set_title('Ground Truth Anomalies')

    axs[0, 2].imshow(Y[:, :, slice_idx], cmap='gray')
    axs[0, 2].set_title('Observed Tensor (Y)')

    # Second row
    # Reconstruct background using the model
    background = tl.tucker_to_tensor((model.C_, model.U_))
    axs[1, 0].imshow(background[:, :, slice_idx], cmap='gray')
    axs[1, 0].set_title('Reconstructed Background')

    axs[1, 1].imshow(A_recovered[:, :, slice_idx], cmap='hot')
    axs[1, 1].set_title('Recovered Anomalies')

    axs[1, 2].imshow(A_pred[:, :, slice_idx], cmap='hot')
    axs[1, 2].set_title('Predicted Anomalies (Binary)')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    tensor_shape = (30, 30, 30)
    tucker_rank = (10, 10, 10)  # Increased rank for better reconstruction
    sparsity = 0.05  # 5% anomalies
    lambda_param = 0.1
    max_iter = 100
    tol = 1e-4
    random_state = 42

    # Step 1: Create Low-Rank Tensor Background (X)
    X, core_true, factors_true = create_low_rank_tensor(
        shape=tensor_shape,
        rank=tucker_rank,
        random_state=random_state
    )

    # Step 2: Create Sparse Anomaly Tensor (A)
    A_true, A_binary = create_sparse_anomaly(
        shape=tensor_shape,
        sparsity=sparsity,
        random_state=random_state
    )

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.01, tensor_shape)
    
    # Generate Observed Tensor (Y = X + A + noise)
    Y = X + A_true + noise

    # Step 3: Initialize and Fit the TensorAnomalyDetection Model
    from tensor_anomaly_detection import TensorAnomalyDetection

    model = TensorAnomalyDetection(
        rank=tucker_rank,
        lambda_param=lambda_param,
        max_iter=max_iter,
        tol=tol
    )

    # Reshape Y for fitting (add sample dimension)
    Y_reshaped = Y.reshape(1, *tensor_shape)
    model.fit(Y_reshaped)

    # Get the recovered anomaly tensor
    A_recovered = model.A_recovered_

    # Compute dynamic threshold based on the distribution of values in A_recovered
    threshold = np.mean(np.abs(A_recovered)) + 2 * np.std(np.abs(A_recovered))
    A_pred = (np.abs(A_recovered) > threshold).astype(int)

    # Evaluate F1 Score
    f1 = evaluate_f1_score_true_pred(A_binary, A_pred)
    print(f"Anomaly Detection F1 Score: {f1:.4f}")

    # Visualization
    visualize_tensors(X, A_binary, Y, model, A_recovered, A_pred)

if __name__ == "__main__":
    main() 
