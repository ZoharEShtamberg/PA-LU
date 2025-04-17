import numpy as np
from LU import lu_piv, verify_decomposition

def print_matrix(matrix, name, precision=4):
    """Print a matrix with a given name and precision."""
    print(f"{name}:")
    for row in matrix:
        print("[", end=" ")
        for val in row:
            print(f"{val:{precision+6}.{precision}f}", end=" ")
        print("]")
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

def run_example(A, name="Example", display=True):
    """Run LU decomposition on the given matrix and verify the result."""
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print(f"{'#'*60}")
    
    print("Original matrix:")
    print_matrix(A, "A")
    
    # Run LU decomposition
    P, L, U = lu_piv(A, disp=1 if display else 0, print_step_func=print_step_matrices)
    
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

# Run the examples
if __name__ == "__main__":
    print("LU DECOMPOSITION WITH PIVOTING EXAMPLES")
    print("=======================================")
    
    run_example(A1, "Simple 2x2 Matrix")
    
    print("\nPress Enter to continue to the next example...")
    input()
    
    run_example(A2, "Matrix Requiring Pivoting")
    
    print("\nPress Enter to continue to the next example...")
    input()
    
    run_example(A3, "Hilbert Matrix (Ill-Conditioned)")
    
    print("\nPress Enter to continue to the next example...")
    input()
    
    run_example(A4, "Block Matrix with Zeros")
    
    print("\nPress Enter to continue to the next example...")
    input()
    
    run_example(A5, "Tridiagonal Matrix")
    
    print("\nAll examples completed. Here's what we've learned:")
    print("1. Simple Matrix: Straightforward decomposition without pivoting needed.")
    print("2. Matrix with Pivoting: Demonstrated how pivoting improves numerical stability.")
    print("3. Hilbert Matrix: Showed how LU decomposition handles ill-conditioned matrices.")
    print("4. Block Matrix: Revealed how zero patterns affect pivoting strategy.")
    print("5. Tridiagonal Matrix: Common in numerical methods for differential equations.")
