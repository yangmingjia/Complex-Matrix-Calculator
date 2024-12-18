#ifndef MATRIX_H
#define MATRIX_H

// Matrix element type
typedef double Element;

// Matrix structure
typedef struct 
{
    int rows; // Number of rows
    int cols; // Number of columns
    Element** data; // 2D array of real numbers
} Matrix;

// Functions for matrix initialization and cleanup
Matrix* create_matrix(int rows, int cols);        // Create new matrix
void free_matrix(Matrix* m);                      // Free matrix memory
Matrix* create_matrix_element(Element value);     // Create 1x1 matrix
Matrix* add_element(Matrix* m, Element value);    // Add element to matrix
Matrix* append_row(Matrix* m1, Matrix* m2);       // Append row to matrix

// Matrix operations
Matrix* matrix_add(Matrix* m1, Matrix* m2);       // Matrix addition 
Matrix* matrix_subtract(Matrix* m1, Matrix* m2);  // Matrix subtraction 
Matrix* matrix_multiply(Matrix* m1, Matrix* m2);  // Matrix multiplication
Matrix* matrix_divide(Matrix* m1, Matrix* m2);    // Matrix division (A/B = A*B^(-1))
Matrix* matrix_power(Matrix* m, Matrix* exp);     // Matrix power
Matrix* matrix_transpose(Matrix* m);              // Matrix transpose
Matrix* matrix_determinant(Matrix* m);            // Compute matrix determinant
Matrix* matrix_inverse(Matrix* m);                // Compute matrix inverse
Matrix* matrix_trace(Matrix* m);                  // Compute matrix trace
Matrix* matrix_rank(Matrix* m);                   // Compute matrix rank
Matrix* matrix_eigenvalues(Matrix* m);            // Compute eigenvalues
Matrix* matrix_eigenvectors(Matrix* m);           // Compute eigenvectors
Matrix* matrix_lu_decomposition(Matrix* m);       // LU decomposition
Matrix* matrix_qr_decomposition(Matrix* m);       // QR decomposition

void print_matrix(Matrix* m);                     // Print matrix
int matrix_is_square(Matrix* m);                  // Check if matrix is square


// Error codes def
typedef enum 
{
    MATRIX_SUCCESS = 0,       // Operation completed
    MATRIX_NULL_PTR,          // Invalid matrix pointer
    MATRIX_DIM_MISMATCH,      // Matrix dimensions do not match
    MATRIX_NOT_SQUARE,        // Matrix is not square
    MATRIX_SINGULAR,          // Matrix is non-invertible
    MATRIX_MEM_ERROR,         // Memory allocation failed
    MATRIX_INVALID_POWER,     // Invalid power operation
    MATRIX_CONVERGENCE_ERROR  // Failed to converge
} MatrixError;

// Error handling functions
void matrix_perror(const char* op, MatrixError err);  // Print error message
MatrixError get_last_error(void); // Get last error code
void set_last_error(MatrixError err); // Set error code

#endif 