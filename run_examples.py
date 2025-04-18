import numpy as np
import argparse
from LU import lu_piv, verify_decomposition

def print_matrix(matrix, name, precision=4):
    """Print a matrix with a given name and precision."""
    # For large matrices, print a condensed preview instead
    if min(matrix.shape) > 15:
        print_large_matrix_preview(matrix, name, precision)
        return
        
    print(f"{name}:")
    for row in matrix:
        print("[", end=" ")
        for val in row:
            print(f"{val:{precision+6}.{precision}f}", end=" ")
        print("]")
    print()

def print_large_matrix_preview(matrix, name, precision=4):
    """Print a condensed preview of a large matrix."""
    rows, cols = matrix.shape
    print(f"{name} (size {rows}×{cols}, showing corner preview):")
    
    # Display the top-left 3x3 corner
    corner_size = min(3, rows, cols)
    print("Top-left corner:")
    for i in range(corner_size):
        print("[", end=" ")
        for j in range(corner_size):
            print(f"{matrix[i,j]:{precision+6}.{precision}f}", end=" ")
        if cols > corner_size:
            print("... ", end="")
        print("]")
    if rows > corner_size:
        print("[", " "*5, "...", " "*5, "]")
    
    # Add some basic statistics
    print(f"Matrix statistics: min={np.min(matrix):.4f}, max={np.max(matrix):.4f}, mean={np.mean(matrix):.4f}")
    print()

def print_step_matrices(L, U, P, A, step):
    """Print the L, U, P, and A matrices at each step."""
    print("\n" + "="*50)
    print(f"Step {step+1} of LU Decomposition")
    print("="*50)
    
    print_matrix(L, "L matrix", precision=4)
    print_matrix(U, "U matrix", precision=4)
    print_matrix(P, "P matrix", precision=1)
    print_matrix(A, "Current A matrix", precision=4)
    
    print("-"*50)
    input("Press Enter to continue to the next step...")

def print_final_result(L, U, P, A, step):
    """Print only the final matrices without waiting for user input."""
    # This function doesn't print anything during steps
    # The final result will be printed by run_example
    pass

def run_example(A, name="Example", display=True, interactive=False):
    """Run LU decomposition on the given matrix and verify the result.
    
    Args:
        A: Input matrix
        name: Example name
        display: Whether to display intermediate steps
        interactive: If True, use interactive print function with user prompts
    """
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print(f"{'#'*60}")
    
    print("Original matrix:")
    print_matrix(A, "A")
    
    # Choose the appropriate print function based on interactive mode
    print_func = print_step_matrices if interactive else print_final_result
    
    # Run LU decomposition
    P, L, U = lu_piv(A, disp=1 if display else 0, print_step_func=print_func)
    
    # Print final results
    print("\nFinal decomposition:")
    print_matrix(L, "L matrix")
    print_matrix(U, "U matrix")
    print_matrix(P, "P matrix")
    
    # Verification
    error = verify_decomposition(A, P, L, U)
    print(f"Verification: ||P*A - L*U|| = {error:.2e}")
    
    # Calculate and display reconstructed matrix
    reconstructed = L @ U
    print("\nReconstructed P*A:")
    print_matrix(reconstructed, "L*U")
    
    return P, L, U

def run_large_example(A, name="Large Example"):
    """Run LU decomposition on a large matrix but only print the error norm.
    
    For very large matrices, printing the full matrices is impractical and
    overwhelming. This function only prints the verification error.
    
    Args:
        A: Input matrix
        name: Example name
    
    Returns:
        P, L, U: The decomposition matrices
    """
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print(f"{'#'*60}")
    
    print(f"Matrix size: {A.shape[0]}×{A.shape[1]}")
    
    # Run LU decomposition without displaying intermediate steps
    P, L, U = lu_piv(A, disp=0)
    
    # Only print verification error
    error = verify_decomposition(A, P, L, U)
    print(f"Verification: ||P*A - L*U|| = {error:.2e}")
    
    return P, L, U

# Example 1: Simple matrix that won't need pivoting
A1 = np.array([
    [4.0, 3.0],
    [6.0, 3.0]
])

# Example 2: Matrix that requires pivoting for stability
A2 = np.array([
    [0.1, 7.0, 2.0],
    [3.0, 4.0, 5.0],
    [8.0, 1.0, 6.0]
])

# Example 3: Hilbert matrix (notoriously ill-conditioned)
def hilbert_matrix(n):
    """Create a Hilbert matrix of size n×n."""
    # Hilbert matrices have elements H[i,j] = 1/(i+j+1)
    # They are famous for being extremely ill-conditioned
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H

A3 = hilbert_matrix(4)  # Using smaller size for readability

# Example 4: Block matrix with zeros - challenges pivoting
A4 = np.array([
    [0.0, 0.0, 2.0, 1.0],
    [0.0, 0.0, 1.0, 2.0],
    [3.0, 1.0, 0.0, 0.0],
    [1.0, 4.0, 0.0, 0.0]
])

# Example 5: Tridiagonal matrix - common in differential equations
n = 5  # Using smaller size for readability
# Creating a tridiagonal matrix with 2 on main diagonal and -1 on sub/super diagonals
# This structure arises when discretizing certain differential equations
A5 = np.zeros((n, n))
for i in range(n):
    A5[i, i] = 2
    if i > 0:
        A5[i, i-1] = -1
    if i < n-1:
        A5[i, i+1] = -1

# Example 6: A large banded matrix - testing performance with bigger matrices
def create_large_banded_matrix(n, bandwidth=5):
    """Create a large banded matrix with the given bandwidth.
    
    Creates an n×n matrix with non-zero elements within a band around the diagonal.
    The matrix has interesting values that ensure it's not singular.
    """
    large_matrix = np.zeros((n, n))
    for i in range(n):
        large_matrix[i, i] = 4.0  # Main diagonal
        
        # Fill bands above the main diagonal
        for k in range(1, bandwidth + 1):
            if i + k < n:
                large_matrix[i, i + k] = 1.0 / (k + 1)  # Decreasing values away from diagonal
        
        # Fill bands below the main diagonal
        for k in range(1, bandwidth + 1):
            if i - k >= 0:
                large_matrix[i, i - k] = -0.5 / (k + 1)  # Negative decreasing values
                
    return large_matrix

# Create a 50x50 matrix with bandwidth 5
A6 = create_large_banded_matrix(50, bandwidth=5)

# Example 7: Toeplitz matrix - matrices with constant diagonals
def create_toeplitz_matrix(n):
    """Create a Toeplitz matrix of size n×n.
    
    A Toeplitz matrix has constant values along all diagonals.
    These matrices arise in signal processing and differential equations.
    """
    toeplitz = np.zeros((n, n))
    
    # Fill the matrix with values that depend only on the difference i-j
    for i in range(n):
        for j in range(n):
            # Main diagonal is 2, and values decrease as we move away
            diagonal_index = i - j
            if diagonal_index == 0:
                toeplitz[i, j] = 2.0  # Main diagonal
            else:
                # Use reciprocal of absolute diagonal index with alternating signs
                toeplitz[i, j] = ((-1) ** abs(diagonal_index)) / (1 + abs(diagonal_index))
    
    return toeplitz

# Create an 8x8 Toeplitz matrix
A7 = create_toeplitz_matrix(8)

# Example 8: Singular matrix - matrix with no inverse (det = 0)
def create_singular_matrix(n=4):
    """Create a singular n×n matrix.
    
    A singular matrix has determinant zero and no inverse.
    In this case, we create a matrix where one row is a linear combination of others.
    """
    # Start with an identity matrix
    singular = np.eye(n)
    
    # Make the last row a linear combination of the other rows
    # This guarantees the matrix will be singular
    singular[-1, :] = 0
    for i in range(n-1):
        singular[-1, :] += singular[i, :] * (i + 1) / (n - 1)
    
    return singular

# Create a 4x4 singular matrix
A8 = create_singular_matrix(4)

# Example 9: Very large random matrix - testing performance with extremely large systems
def create_large_random_matrix(n, seed=42):
    """Create a large n×n matrix with random values.
    
    This is useful for performance testing and demonstrating how LU decomposition
    scales with very large matrices.
    
    Args:
        n: Size of the matrix
        seed: Random seed for reproducibility
    
    Returns:
        A random n×n matrix
    """
    np.random.seed(seed)  # For reproducibility
    return np.random.rand(n, n)

# Run the examples
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run LU decomposition examples")
    parser.add_argument("-i", "--interactive", action="store_true", 
                        help="Run in interactive mode with step-by-step visualization")
    parser.add_argument("-l", "--large-matrix", type=int, metavar="N", 
                        help="Run an additional example with a large random N×N matrix")
    parser.add_argument("-o", "--only-large-matrix", type=int, metavar="N", 
                        help="Only run the large random N×N matrix example")
    args = parser.parse_args()
    
    
    # Use the command-line flag to set interactive mode
    interactive_mode = args.interactive
    
    # Determine if we're running only the large matrix example
    only_large_matrix = args.only_large_matrix is not None
    
    if interactive_mode:
        print("Running in INTERACTIVE mode. You will be prompted at each step.")
    
    # Run standard examples if not in only-large-matrix mode
    if not only_large_matrix:
        run_example(A1, "Simple 2x2 Matrix", interactive=interactive_mode)
        
        if interactive_mode:
            print("\nPress Enter to continue to the next example...")
            input()
        
        run_example(A2, "Matrix Requiring Pivoting", interactive=interactive_mode)
        
        if interactive_mode:
            print("\nPress Enter to continue to the next example...")
            input()
        
        run_example(A3, "Hilbert Matrix (Ill-Conditioned)", interactive=interactive_mode)
        
        if interactive_mode:
            print("\nPress Enter to continue to the next example...")
            input()
        
        run_example(A4, "Block Matrix with Zeros", interactive=interactive_mode)
        
        if interactive_mode:
            print("\nPress Enter to continue to the next example...")
            input()
        
        run_example(A5, "Tridiagonal Matrix", interactive=interactive_mode)
        
        if interactive_mode:
            print("\nPress Enter to continue to the next example...")
            input()
        
        run_example(A6, "Large Banded Matrix (50×50)", display=True, interactive=False)
        
        if interactive_mode:
            print("\nPress Enter to continue to the next example...")
            input()
        
        # Add the Toeplitz matrix example
        run_example(A7, "Toeplitz Matrix", interactive=interactive_mode)
        
        if interactive_mode:
            print("\nPress Enter to continue to the next example...")
            input()
            
        # Add the singular matrix example
        run_example(A8, "Singular Matrix", interactive=interactive_mode)
        
        if interactive_mode and (args.large_matrix or args.only_large_matrix):
            print("\nPress Enter to continue to the large matrix example...")
            input()

    # Run the large random matrix example if specified
    large_matrix_size = args.large_matrix or args.only_large_matrix
    if large_matrix_size:
        print(f"\nNow running LU decomposition on a {large_matrix_size}×{large_matrix_size} random matrix...")
        print("This may take some time.")
        large_random_matrix = create_large_random_matrix(large_matrix_size)
        run_large_example(large_random_matrix, f"Large Random Matrix ({large_matrix_size}×{large_matrix_size})")
    
    if not only_large_matrix:
        print("\nAll examples completed. Here's what we've learned:")
        print("1. Simple Matrix: Straightforward decomposition without pivoting needed.")
        print("2. Matrix with Pivoting: Demonstrated how pivoting improves numerical stability.")
        print("3. Hilbert Matrix: Showed how LU decomposition handles ill-conditioned matrices.")
        print("4. Block Matrix: Revealed how zero patterns affect pivoting strategy.")
        print("5. Tridiagonal Matrix: Common in numerical methods for differential equations.")
        print("6. Large Banded Matrix: Demonstrated performance on larger systems.")
        print("7. Toeplitz Matrix: Illustrated decomposition of matrices with constant diagonals.")
        print("8. Singular Matrix: Showed how decomposition behaves with non-invertible matrices.")
        if large_matrix_size:
            print(f"9. Large Random Matrix: Showed scaling behavior with extremely large systems ({large_matrix_size}×{large_matrix_size}).")
