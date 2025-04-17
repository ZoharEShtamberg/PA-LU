# vibe coded this for educational purposes

# LU Decomposition with Pivoting

This repository contains a Python implementation of LU decomposition with partial pivoting, designed for educational purposes. It visualizes the step-by-step process of LU decomposition and includes several examples with different types of matrices.

## About LU Decomposition

LU decomposition is a numerical method that factors a matrix as the product of a lower triangular matrix (L) and an upper triangular matrix (U). When pivoting is included (as in this implementation), we have:

PA = LU

Where:
- P is a permutation matrix
- L is a lower triangular matrix with ones on the diagonal
- U is an upper triangular matrix

This decomposition is useful for:
- Solving systems of linear equations
- Computing determinants
- Matrix inversion
- And more!

## Usage

Run the examples with:

```bash
python run_examples.py
```

### Command Line Options

The script supports several command line options:

- **Interactive Mode**:
  ```bash
  python run_examples.py -i
  ```
  or
  ```bash
  python run_examples.py --interactive
  ```
  This mode provides step-by-step visualization of the LU decomposition process with pauses after each step.

- **Include a Large Random Matrix**:
  ```bash
  python run_examples.py -l 100
  ```
  or
  ```bash
  python run_examples.py --large-matrix 100
  ```
  Runs all standard examples plus an additional example with a large random matrix of size N×N (100×100 in this example).

- **Run Only the Large Matrix Example**:
  ```bash
  python run_examples.py -o 200
  ```
  or
  ```bash
  python run_examples.py --only-large-matrix 200
  ```
  Skips all standard examples and only runs the large random matrix example of size N×N (200×200 in this example).

- **Combine Options**:
  ```bash
  python run_examples.py -i -l 150
  ```
  Run in interactive mode and include a 150×150 random matrix example.

The interactive mode (`-i` flag) provides:
- Step-by-step visualization of the LU decomposition process
- Pauses after each step so you can examine the matrices
- Prompts to continue to the next step or next example
- Detailed inspection of how the algorithm works

This is especially useful for educational purposes to understand how LU decomposition transforms the original matrix into its factorized form.

## Examples Included

The repository includes examples demonstrating LU decomposition on various matrices:

1. **Simple Matrix**: Straightforward decomposition without pivoting needed
2. **Matrix Requiring Pivoting**: Demonstrates how pivoting improves numerical stability
3. **Hilbert Matrix**: Shows how LU decomposition handles ill-conditioned matrices
4. **Block Matrix with Zeros**: Reveals how zero patterns affect pivoting strategy
5. **Tridiagonal Matrix**: Common in numerical methods for differential equations
6. **Large Banded Matrix**: Demonstrates performance on larger systems (50×50)
7. **Toeplitz Matrix**: Illustrates decomposition of matrices with constant diagonals
8. **Large Random Matrix**: Tests the algorithm on a large randomly generated matrix

Each example verifies the decomposition by computing the error between PA and LU.
