import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker
from tensorly.tenalg.proximal import soft_thresholding

def three_way_decomposition(Y, lambda_param=0.1, max_iter=100, tol=1e-6):
    """
    Perform three-way decomposition of tensor Y into background (X), anomaly (A), and noise (E).
    
    Parameters:
    - Y: Input tensor (observed data)
    - lambda_param: Regularization parameter for anomaly component
    - max_iter: Maximum number of iterations for optimization
    - tol: Tolerance for convergence
    
    Returns:
    - C: Core tensor from Tucker decomposition
    - U: List of factor matrices from Tucker decomposition
    - A: Anomaly tensor
    """
    # Initialize A as zeros
    A = np.zeros_like(Y)
    
    # Initialize U by performing Tucker decomposition on Y
    C, U = non_negative_tucker(Y, rank=[min(Y.shape) // 2] * len(Y.shape), n_iter_max=100, tol=1e-6)
    
    for iteration in range(max_iter):
        # Step 1: Optimize U given A
        Y_minus_A = Y - A
        C, U = non_negative_tucker(Y_minus_A, rank=[min(Y.shape) // 2] * len(Y.shape), n_iter_max=100, tol=1e-6)
        
        # Step 2: Optimize A given U
        # Compute the reconstructed tensor
        reconstructed = tl.tucker_to_tensor((C, U))
        
        # Update A using soft thresholding
        A_new = soft_thresholding(Y - reconstructed, lambda_param)
        
        # Check for convergence
        if tl.norm(A_new - A) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
        
        A = A_new
    
    return C, U, A

if __name__ == "__main__":
    # Example usage
    # Generate synthetic tensor data
    np.random.seed(42)
    Y = np.random.rand(10, 10, 10)  # Example tensor
    
    # Perform three-way decomposition
    C, U, A = three_way_decomposition(Y, lambda_param=0.1)
    
    print("Core tensor C shape:", C.shape)
    print("Factor matrices U shapes:", [u.shape for u in U])
    print("Anomaly tensor A shape:", A.shape) 