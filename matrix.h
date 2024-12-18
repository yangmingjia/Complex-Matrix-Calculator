#ifndef MATRIX_H
#define MATRIX_H

/* Matrix element type */
typedef double Element;

/* Matrix structure
 * Represents a matrix of real numbers
 * Stores dimensions and a 2D array of real numbers
 */
typedef struct 
{
    int rows;       /* Number of rows in the matrix */
    int cols;       /* Number of columns in the matrix */
    Element** data; /* 2D array of real numbers */
} Matrix;

/* Matrix creation and memory management
 * Functions for matrix initialization and cleanup
 */
Matrix* create_matrix(int rows, int cols);        /* Create new matrix */
void free_matrix(Matrix* m);                      /* Free matrix memory */
Matrix* create_matrix_element(Element value);     /* Create 1x1 matrix */
Matrix* add_element(Matrix* m, Element value);    /* Add element to matrix */
Matrix* append_row(Matrix* m1, Matrix* m2);       /* Append row to matrix */

/* Basic matrix operations
 * Fundamental matrix arithmetic and operations
 */
Matrix* matrix_add(Matrix* m1, Matrix* m2);       /* Matrix addition */
Matrix* matrix_subtract(Matrix* m1, Matrix* m2);  /* Matrix subtraction */
Matrix* matrix_multiply(Matrix* m1, Matrix* m2);  /* Matrix multiplication */
Matrix* matrix_transpose(Matrix* m);              /* Matrix transpose */
Matrix* matrix_determinant(Matrix* m);            /* Matrix determinant */

/* Advanced matrix operations
 * Higher-level matrix computations
 */
Matrix* matrix_inverse(Matrix* m);                /* Matrix inverse */
Matrix* matrix_eigenvalues(Matrix* m);            /* Compute eigenvalues */
Matrix* matrix_eigenvectors(Matrix* m);           /* Compute eigenvectors */
Matrix* matrix_lu_decomposition(Matrix* m);       /* LU decomposition */
Matrix* matrix_qr_decomposition(Matrix* m);       /* QR decomposition */
Matrix* matrix_schur_decomposition(Matrix* m);    /* Schur decomposition */

/* Additional matrix operations
 * Extended functionality for matrix computations
 */
Matrix* matrix_divide(Matrix* m1, Matrix* m2);    /* Matrix division (A/B = A*B^(-1)) */
Matrix* matrix_power(Matrix* m, Matrix* exp);     /* Matrix power */
Matrix* solve_linear_system(Matrix* A, Matrix* b); /* Solve Ax = b */

/* Advanced matrix functions
 * Matrix exponential, logarithm and norm
 */
Matrix* matrix_exp(Matrix* m);                    /* Matrix exponential */
Matrix* matrix_log(Matrix* m);                  /* Matrix logarithm */
double matrix_norm(Matrix* m);                  /* Frobenius norm */

/* Utility functions
 * Helper functions for matrix operations
 */
void print_matrix(Matrix* m);                     /* Print matrix to stdout */
int matrix_is_square(Matrix* m);                  /* Check if matrix is square */
Matrix* matrix_trace(Matrix* m);                  /* Compute matrix trace */
int matrix_rank(Matrix* m);                       /* Compute matrix rank */

/* Error handling
 * Error reporting and management
 */

/* Error code definitions for matrix operations */
typedef enum {
    MATRIX_SUCCESS = 0,        /* Operation completed successfully */
    MATRIX_NULL_PTR,          /* Invalid matrix pointer */
    MATRIX_DIM_MISMATCH,      /* Matrix dimensions do not match */
    MATRIX_NOT_SQUARE,        /* Operation requires square matrix */
    MATRIX_SINGULAR,          /* Matrix is singular (non-invertible) */
    MATRIX_MEM_ERROR,         /* Memory allocation failed */
    MATRIX_INVALID_POWER,     /* Invalid power operation */
    MATRIX_CONVERGENCE_ERROR  /* Algorithm failed to converge */
} MatrixError;

/* Error handling function declarations */
void matrix_perror(const char* op, MatrixError err);  /* Print error message */
MatrixError get_last_error(void);                     /* Get last error code */
void set_last_error(MatrixError err);                 /* Set error code */

#endif 