#ifndef MATRIX_H
#define MATRIX_H

// Complex number structure
typedef struct {
    double real;
    double imag;
} Complex;

// Matrix structure with complex numbers
typedef struct {
    int rows;
    int cols;
    Complex** data;
} Matrix;

// Complex number operations
Complex complex_add(Complex a, Complex b);
Complex complex_subtract(Complex a, Complex b);
Complex complex_multiply(Complex a, Complex b);
Complex complex_divide(Complex a, Complex b);
Complex complex_conjugate(Complex a);
double complex_abs(Complex a);
Complex complex_from_real(double real);

// Matrix creation and memory management
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix* m);
Matrix* create_matrix_element(Complex value);
Matrix* add_element(Matrix* m, Complex value);
Matrix* append_row(Matrix* m1, Matrix* m2);

// Basic matrix operations
Matrix* matrix_add(Matrix* m1, Matrix* m2);
Matrix* matrix_subtract(Matrix* m1, Matrix* m2);
Matrix* matrix_multiply(Matrix* m1, Matrix* m2);
Matrix* matrix_transpose(Matrix* m);
Matrix* matrix_conjugate_transpose(Matrix* m);
Matrix* matrix_determinant(Matrix* m);

// Advanced matrix operations
Matrix* matrix_inverse(Matrix* m);
Matrix* matrix_eigenvalues(Matrix* m);
Matrix* matrix_eigenvectors(Matrix* m);
Matrix* matrix_lu_decomposition(Matrix* m);
Matrix* matrix_qr_decomposition(Matrix* m);
Matrix* matrix_schur_decomposition(Matrix* m);

// Additional matrix operations
Matrix* matrix_divide(Matrix* m1, Matrix* m2);
Matrix* matrix_power(Matrix* m1, Matrix* m2);
Matrix* solve_linear_system(Matrix* A, Matrix* b);
Complex complex_sqrt(Complex a);

// Utility functions
void print_matrix(Matrix* m);
int matrix_is_square(Matrix* m);
Complex matrix_trace(Matrix* m);
int matrix_rank(Matrix* m);
int matrix_is_hermitian(Matrix* m);

// Error handling
extern int matrix_errno;
const char* matrix_strerror(int errno);

#endif 