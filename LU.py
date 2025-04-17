import numpy as np
from typing import Tuple, Any
from numpy.typing import NDArray




# Print_step_func type: Optional[Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int], None]]
def lu_piv(A: NDArray[np.float64], disp: int = 1, 
           print_step_func = None) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """LU decomposition with partial pivoting: P*A = L*U
    
    Parameters:
        A (numpy.ndarray): Input matrix to decompose
        disp (int): If 1, display matrices at each step
        print_step_func (function): Function to print the intermediate steps of the decomposition.
            Called with arguments (L, U, P, A, k) at each iteration k, where:
            - L: Current lower triangular matrix
            - U: Current upper triangular matrix
            - P: Current permutation matrix
            - A: Working copy of the matrix being factorized
            - k: Current iteration number (0-based)
        
    Returns:
        P (numpy.ndarray): Permutation matrix
        L (numpy.ndarray): Lower triangular matrix
        U (numpy.ndarray): Upper triangular matrix
    """
    # Get matrix dimensions
    m, n = A.shape
    
    # Initialize matrices
    P = np.eye(m)  # Permutation matrix (identity to start)
    L = np.eye(m)  # Lower triangular matrix (identity to start)
    U = np.zeros((m, n))  # Upper triangular matrix (zeros to start)
    
    # Numerical tolerance (equivalent to MATLAB's sqrt(eps))
    tol = np.sqrt(np.finfo(float).eps)
    
    # Create a copy of A to work with
    A = A.copy()
    
    # Main decomposition loop
    for k in range(min(m, n)):
        # ARRAY SLICING: A[k:m, k]
        # - Selects rows k through m-1 (all rows from k to the end)
        # - And specifically column k
        # - This gives us the sub-column below and including the diagonal element
        r = find_pivot(A[k:m, k], k)
        
        # Swap rows if needed
        if r != k:
            # ROW SWAPPING: A[[r, k], :] = A[[k, r], :]
            # - NumPy allows using lists of indices to select multiple rows/columns
            # - A[[r, k], :] selects rows r and k (in that order) and all columns
            # - A[[k, r], :] selects rows k and r (in that order) and all columns
            # - The assignment swaps these rows in a single operation
            # - This is much more efficient than using a temporary variable
            A[[r, k], :] = A[[k, r], :]
            
            # PARTIAL ROW SWAPPING: L[[r, k], :k]
            # - Similar to above, but only affecting columns 0 to k-1
            # - This preserves the L matrix structure we've built so far
            # - We only swap the part of L that's already been computed
            L[[r, k], :k] = L[[k, r], :k]
            
            # Swap rows in P to track the permutations
            P[[r, k], :] = P[[k, r], :]
        
        # ARRAY SLICE ASSIGNMENT: U[k, k:n] = A[k, k:n]
        # - Copies row k, columns k through n-1 from A to U
        # - This populates one row of the upper triangular matrix
        U[k, k:n] = A[k, k:n]
        
        # Check if pivot is large enough
        if abs(A[k, k]) >= tol:
            # VECTORIZED DIVISION: L[k+1:m, k] = A[k+1:m, k] / A[k, k]
            # - This divides each element of the vector A[k+1:m, k] by the scalar A[k, k]
            # - Creates the multipliers needed for elimination
            # - NumPy automatically broadcasts the scalar division to each element
            L[k+1:m, k] = A[k+1:m, k] / A[k, k]
            
            # OUTER PRODUCT AND SUBMATRIX UPDATE:
            # A[k+1:m, k+1:n] = A[k+1:m, k+1:n] - np.outer(L[k+1:m, k], U[k, k+1:n])
            #
            # What is an outer product?
            # - For vectors a (mx1) and b (1xn), the outer product a⊗b creates an mxn matrix
            # - Each element (i,j) equals a[i] × b[j]
            # - np.outer(a, b) efficiently computes this matrix
            #
            # In this context:
            # - L[k+1:m, k] is a column vector of multipliers (mx1)
            # - U[k, k+1:n] is a row vector from the current pivot row (1xn)
            # - Their outer product creates exactly the matrix we need to subtract
            # - This is the elimination step that updates the remaining submatrix
            # - It's equivalent to applying all row operations at once
            A[k+1:m, k+1:n] = A[k+1:m, k+1:n] - np.outer(L[k+1:m, k], U[k, k+1:n])
        else:
            print(f"Warning: Small pivot encountered at step {k}, matrix may be singular.")
            return P, L, U
        
        if disp == 1 and print_step_func is not None:
            print_step_func(L, U, P, A, k)
    
    return P, L, U

def find_pivot(a: NDArray[np.float64], k: int) -> int:
    """Find the position of the maximum absolute value element."""
    # np.argmax returns the index of the maximum value in the array
    pos = np.argmax(abs(a))
    # Add k to get the actual row index in the original matrix
    return pos + k

def verify_decomposition(A: NDArray[np.float64], P: NDArray[np.float64], L: NDArray[np.float64], U: NDArray[np.float64]) -> float:
    """Verify that P*A = L*U and return the error."""
    # @ is the matrix multiplication operator in NumPy
    # np.linalg.norm computes the Frobenius norm of the difference matrix
    return np.linalg.norm(P @ A - L @ U)
