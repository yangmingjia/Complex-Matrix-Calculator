#include "matrix.h"
#include "parser.tab.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Declare yyparse function
int yyparse(void);

// Creates a new matrix with given dimensions
Matrix* create_matrix(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (Complex**)malloc(rows * sizeof(Complex*));
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    for (int i = 0; i < rows; i++) {
        m->data[i] = (Complex*)calloc(cols, sizeof(Complex));
        if (!m->data[i]) {
            for (int j = 0; j < i; j++) {
                free(m->data[j]);
            }
            free(m->data);
            free(m);
            return NULL;
        }
    }
    return m;
}

// Creates a 1x1 matrix with given value
Matrix* create_matrix_element(Complex value) {
    Matrix* m = create_matrix(1, 1);
    if (m) m->data[0][0] = value;
    return m;
}

// Adds a new element to the matrix row
Matrix* add_element(Matrix* m, Complex value) {
    if (!m) return NULL;
    
    Matrix* new_m = create_matrix(m->rows, m->cols + 1);
    if (!new_m) return NULL;
    
    // Copy existing elements
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            new_m->data[i][j] = m->data[i][j];
        }
    }
    
    // Add new element
    new_m->data[0][m->cols] = value;
    free_matrix(m);
    return new_m;
}

// Appends a row to the matrix
Matrix* append_row(Matrix* m1, Matrix* m2) {
    if (!m1 || !m2 || m1->cols != m2->cols) return NULL;
    
    Matrix* new_m = create_matrix(m1->rows + 1, m1->cols);
    if (!new_m) return NULL;
    
    // Copy existing rows
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            new_m->data[i][j] = m1->data[i][j];
        }
    }
    
    // Add new row
    for (int j = 0; j < m2->cols; j++) {
        new_m->data[m1->rows][j] = m2->data[0][j];
    }
    
    free_matrix(m1);
    free_matrix(m2);
    return new_m;
}

// Matrix addition
Matrix* matrix_add(Matrix* m1, Matrix* m2) {
    if (!m1 || !m2 || m1->rows != m2->rows || m1->cols != m2->cols) return NULL;
    
    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = complex_add(m1->data[i][j], m2->data[i][j]);
        }
    }
    return result;
}

// Matrix subtraction
Matrix* matrix_subtract(Matrix* m1, Matrix* m2) {
    if (!m1 || !m2 || m1->rows != m2->rows || m1->cols != m2->cols) return NULL;
    
    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = complex_subtract(m1->data[i][j], m2->data[i][j]);
        }
    }
    return result;
}

// Matrix multiplication
Matrix* matrix_multiply(Matrix* m1, Matrix* m2) {
    if (!m1 || !m2 || m1->cols != m2->rows) return NULL;
    
    Matrix* result = create_matrix(m1->rows, m2->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            result->data[i][j].real = 0;
            result->data[i][j].imag = 0;
            for (int k = 0; k < m1->cols; k++) {
                result->data[i][j] = complex_add(result->data[i][j], 
                    complex_multiply(m1->data[i][k], m2->data[k][j]));
            }
        }
    }
    return result;
}

// Matrix transpose
Matrix* matrix_transpose(Matrix* m) {
    if (!m) return NULL;
    
    Matrix* result = create_matrix(m->cols, m->rows);
    if (!result) return NULL;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j][i] = m->data[i][j];
        }
    }
    return result;
}

// Calculate determinant using recursive method
Complex calc_determinant(Matrix* m) {
    if (m->rows != m->cols) return (Complex){0, 0};
    if (m->rows == 1) return m->data[0][0];
    if (m->rows == 2) {
        Complex det = {
            .real = m->data[0][0].real * m->data[1][1].real - m->data[0][1].real * m->data[1][0].real,
            .imag = m->data[0][0].imag * m->data[1][1].imag - m->data[0][1].imag * m->data[1][0].imag
        };
        return det;
    }
    
    Complex det = {0, 0};
    int sign = 1;
    
    for (int j = 0; j < m->cols; j++) {
        Matrix* submatrix = create_matrix(m->rows - 1, m->cols - 1);
        if (!submatrix) return (Complex){0, 0};
        
        // Create submatrix
        for (int row = 1; row < m->rows; row++) {
            int sub_col = 0;
            for (int col = 0; col < m->cols; col++) {
                if (col == j) continue;
                submatrix->data[row-1][sub_col] = m->data[row][col];
                sub_col++;
            }
        }
        
        Complex sub_det = calc_determinant(submatrix);
        det = complex_add(det, complex_multiply(complex_multiply(m->data[0][j], sub_det), 
            (Complex){sign, 0}));
        sign = -sign;
        free_matrix(submatrix);
    }
    
    return det;
}

// Matrix determinant wrapper function
Matrix* matrix_determinant(Matrix* m) {
    if (!m || m->rows != m->cols) return NULL;
    Complex det = calc_determinant(m);
    return create_matrix_element(det);
}

// Free matrix memory
void free_matrix(Matrix* m) {
    if (!m) return;
    
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Check if a complex number is essentially real (imaginary part near zero)
static int is_real_number(Complex c) {
    return fabs(c.imag) < 1e-10;
}

// Check if matrix contains only real numbers
static int is_real_matrix(Matrix* m) {
    if (!m) return 0;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (!is_real_number(m->data[i][j])) {
                return 0;
            }
        }
    }
    return 1;
}

// Print matrix
void print_matrix(Matrix* m) {
    if (!m) return;
    
    // Check if matrix is real
    int is_real = is_real_matrix(m);
    
    // First pass: find maximum width needed for each column
    int* col_widths = (int*)calloc(m->cols, sizeof(int));
    if (!col_widths) return;
    
    for (int j = 0; j < m->cols; j++) {
        for (int i = 0; i < m->rows; i++) {
            char buf[50];
            if (is_real) {
                double val = m->data[i][j].real;
                if (fabs(val - round(val)) < 1e-10) {
                    snprintf(buf, sizeof(buf), "%.0f", val);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f", val);
                }
            } else {
                double real = m->data[i][j].real;
                double imag = m->data[i][j].imag;
                char real_buf[25], imag_buf[25];
                
                if (fabs(real - round(real)) < 1e-10) {
                    snprintf(real_buf, sizeof(real_buf), "%.0f", real);
                } else {
                    snprintf(real_buf, sizeof(real_buf), "%.2f", real);
                }
                
                if (fabs(imag - round(imag)) < 1e-10) {
                    snprintf(imag_buf, sizeof(imag_buf), "%.0f", imag);
                } else {
                    snprintf(imag_buf, sizeof(imag_buf), "%.2f", imag);
                }
                
                snprintf(buf, sizeof(buf), "(%s, %s)", real_buf, imag_buf);
            }
            int len = strlen(buf);
            if (len > col_widths[j]) {
                col_widths[j] = len;
            }
        }
    }
    
    printf("[\n");
    for (int i = 0; i < m->rows; i++) {
        printf("  ");
        for (int j = 0; j < m->cols; j++) {
            if (is_real) {
                double val = m->data[i][j].real;
                if (fabs(val - round(val)) < 1e-10) {
                    printf("%*.*f", col_widths[j], 0, val);
                } else {
                    printf("%*.*f", col_widths[j], 2, val);
                }
            } else {
                double real = m->data[i][j].real;
                double imag = m->data[i][j].imag;
                char real_buf[25], imag_buf[25];
                
                if (fabs(real - round(real)) < 1e-10) {
                    snprintf(real_buf, sizeof(real_buf), "%.0f", real);
                } else {
                    snprintf(real_buf, sizeof(real_buf), "%.2f", real);
                }
                
                if (fabs(imag - round(imag)) < 1e-10) {
                    snprintf(imag_buf, sizeof(imag_buf), "%.0f", imag);
                } else {
                    snprintf(imag_buf, sizeof(imag_buf), "%.2f", imag);
                }
                
                printf("(%s, %s)", real_buf, imag_buf);
                int padding = col_widths[j] - strlen(real_buf) - strlen(imag_buf) - 4;
                if (padding > 0) printf("%*s", padding, "");
            }
            if (j < m->cols - 1) printf("  ");
        }
        printf("\n");
    }
    printf("]\n");
    
    free(col_widths);
}

// Utility function to check if matrix is square
int matrix_is_square(Matrix* m) {
    return m && (m->rows == m->cols);
}

// Calculate matrix trace (sum of diagonal elements)
Complex matrix_trace(Matrix* m) {
    if (!matrix_is_square(m)) return (Complex){0, 0};
    
    Complex trace = {0, 0};
    for (int i = 0; i < m->rows; i++) {
        trace = complex_add(trace, m->data[i][i]);
    }
    return trace;
}

// LU Decomposition using Doolittle's method
Matrix* matrix_lu_decomposition(Matrix* m) {
    if (!matrix_is_square(m)) return NULL;
    
    int n = m->rows;
    Matrix* L = create_matrix(n, n);
    Matrix* U = create_matrix(n, n);
    if (!L || !U) {
        free_matrix(L);
        free_matrix(U);
        return NULL;
    }

    // Initialize L's diagonal to 1
    for (int i = 0; i < n; i++) {
        L->data[i][i] = (Complex){1, 0};
    }

    // Calculate L and U matrices
    for (int j = 0; j < n; j++) {
        // Calculate U's elements
        for (int i = 0; i <= j; i++) {
            Complex sum = {0, 0};
            for (int k = 0; k < i; k++) {
                sum = complex_add(sum, complex_multiply(L->data[i][k], U->data[k][j]));
            }
            U->data[i][j] = complex_subtract(m->data[i][j], sum);
        }

        // Calculate L's elements
        for (int i = j + 1; i < n; i++) {
            Complex sum = {0, 0};
            for (int k = 0; k < j; k++) {
                sum = complex_add(sum, complex_multiply(L->data[i][k], U->data[k][j]));
            }
            if (complex_abs(U->data[j][j]) < 1e-10) {
                free_matrix(L);
                free_matrix(U);
                return NULL; // Matrix is singular
            }
            L->data[i][j] = complex_divide(complex_subtract(m->data[i][j], sum), U->data[j][j]);
        }
    }

    // Combine L and U into a single matrix
    // Store L below diagonal and U above (including) diagonal
    Matrix* result = create_matrix(n, n);
    if (!result) {
        free_matrix(L);
        free_matrix(U);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) {
                result->data[i][j] = L->data[i][j];
            } else {
                result->data[i][j] = U->data[i][j];
            }
        }
    }

    free_matrix(L);
    free_matrix(U);
    return result;
}

// QR Decomposition using Gram-Schmidt process
Matrix* matrix_qr_decomposition(Matrix* m) {
    if (!m || m->rows < m->cols) return NULL;
    
    int rows = m->rows;
    int cols = m->cols;
    
    Matrix* Q = create_matrix(rows, cols);
    Matrix* R = create_matrix(cols, cols);
    if (!Q || !R) {
        free_matrix(Q);
        free_matrix(R);
        return NULL;
    }

    // Perform Gram-Schmidt process
    for (int j = 0; j < cols; j++) {
        // Copy column j of A to Q
        for (int i = 0; i < rows; i++) {
            Q->data[i][j] = m->data[i][j];
        }

        // Orthogonalize
        for (int k = 0; k < j; k++) {
            Complex dot_product = {0, 0};
            for (int i = 0; i < rows; i++) {
                dot_product = complex_add(dot_product, 
                    complex_multiply(Q->data[i][k], m->data[i][j]));
            }
            R->data[k][j] = dot_product;
            
            for (int i = 0; i < rows; i++) {
                Q->data[i][j] = complex_subtract(Q->data[i][j], 
                    complex_multiply(dot_product, Q->data[i][k]));
            }
        }

        // Normalize
        Complex norm = {0, 0};
        for (int i = 0; i < rows; i++) {
            norm = complex_add(norm, complex_multiply(Q->data[i][j], Q->data[i][j]));
        }
        norm = complex_sqrt(norm);
        
        if (complex_abs(norm) < 1e-10) {
            free_matrix(Q);
            free_matrix(R);
            return NULL; // Linearly dependent columns
        }

        R->data[j][j] = norm;
        for (int i = 0; i < rows; i++) {
            Q->data[i][j] = complex_divide(Q->data[i][j], norm);
        }
    }

    // Combine Q and R into a single matrix
    // Store Q in the first rows x cols block
    // Store R in the next cols x cols block
    Matrix* result = create_matrix(rows + cols, cols);
    if (!result) {
        free_matrix(Q);
        free_matrix(R);
        return NULL;
    }

    // Copy Q
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result->data[i][j] = Q->data[i][j];
        }
    }
    
    // Copy R
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            result->data[rows + i][j] = R->data[i][j];
        }
    }

    free_matrix(Q);
    free_matrix(R);
    return result;
}

// Matrix inverse using Gauss-Jordan elimination
Matrix* matrix_inverse(Matrix* m) {
    if (!matrix_is_square(m)) return NULL;
    
    int n = m->rows;
    Matrix* augmented = create_matrix(n, 2 * n);
    if (!augmented) return NULL;

    // Create augmented matrix [A|I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented->data[i][j] = m->data[i][j];
            augmented->data[i][j + n] = (i == j) ? (Complex){1, 0} : (Complex){0, 0};
        }
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        Complex pivot = augmented->data[i][i];
        if (complex_abs(pivot) < 1e-10) {
            free_matrix(augmented);
            return NULL; // Matrix is singular
        }

        // Normalize row i
        for (int j = 0; j < 2 * n; j++) {
            augmented->data[i][j] = complex_divide(augmented->data[i][j], pivot);
        }

        // Eliminate column i
        for (int j = 0; j < n; j++) {
            if (i != j) {
                Complex factor = augmented->data[j][i];
                for (int k = 0; k < 2 * n; k++) {
                    augmented->data[j][k] = complex_subtract(augmented->data[j][k], 
                        complex_multiply(factor, augmented->data[i][k]));
                }
            }
        }
    }

    // Extract inverse matrix
    Matrix* inverse = create_matrix(n, n);
    if (!inverse) {
        free_matrix(augmented);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse->data[i][j] = augmented->data[i][j + n];
        }
    }

    free_matrix(augmented);
    return inverse;
}

// Calculate eigenvalues using QR algorithm
Matrix* matrix_eigenvalues(Matrix* m) {
    if (!matrix_is_square(m)) return NULL;
    
    int n = m->rows;
    Matrix* A = create_matrix(n, n);
    if (!A) return NULL;

    // Copy input matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A->data[i][j] = m->data[i][j];
        }
    }

    // Perform QR iterations with shifts
    const int MAX_ITER = 100;
    const double EPSILON = 1e-10;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Calculate Wilkinson shift
        Complex shift;
        if (n >= 2) {
            Complex a = A->data[n-2][n-2];
            Complex b = A->data[n-2][n-1];
            Complex c = A->data[n-1][n-1];
            Complex d = complex_subtract(a, c);
            Complex t = complex_multiply(d, d);
            t = complex_add(t, complex_multiply(b, b));
            t.real = sqrt(t.real * t.real + t.imag * t.imag);
            
            if (d.real * d.real + d.imag * d.imag > 0) {
                shift = complex_add(c, complex_divide(complex_multiply(b, b),
                    complex_add(d, (Complex){t.real * (d.real >= 0 ? 1 : -1), 0})));
            } else {
                shift = complex_subtract(c, complex_divide(complex_multiply(b, b),
                    complex_add(d, (Complex){t.real * (d.real >= 0 ? 1 : -1), 0})));
            }
        } else {
            shift = A->data[0][0];
        }

        // Shift matrix
        for (int i = 0; i < n; i++) {
            A->data[i][i] = complex_subtract(A->data[i][i], shift);
        }

        // QR decomposition
        Matrix* QR = matrix_qr_decomposition(A);
        if (!QR) {
            free_matrix(A);
            return NULL;
        }

        // Extract Q and R
        Matrix* Q = create_matrix(n, n);
        Matrix* R = create_matrix(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q->data[i][j] = QR->data[i][j];
                R->data[i][j] = QR->data[n + i][j];
            }
        }

        // A = RQ + shift*I
        Matrix* new_A = matrix_multiply(R, Q);
        for (int i = 0; i < n; i++) {
            new_A->data[i][i] = complex_add(new_A->data[i][i], shift);
        }

        // Check convergence
        double diff = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Complex d = complex_subtract(new_A->data[i][j], A->data[i][j]);
                diff += sqrt(d.real * d.real + d.imag * d.imag);
            }
        }

        free_matrix(A);
        A = new_A;
        free_matrix(QR);
        free_matrix(Q);
        free_matrix(R);

        if (diff < EPSILON) break;
    }

    // Extract eigenvalues from diagonal
    Matrix* eigenvalues = create_matrix(n, 1);
    if (!eigenvalues) {
        free_matrix(A);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        eigenvalues->data[i][0] = A->data[i][i];
    }

    free_matrix(A);
    return eigenvalues;
}

// Complex eigenvector calculation
Matrix* matrix_eigenvectors(Matrix* m) {
    if (!matrix_is_square(m)) return NULL;
    
    Matrix* eigenvalues = matrix_eigenvalues(m);
    if (!eigenvalues) return NULL;

    int n = m->rows;
    Matrix* eigenvectors = create_matrix(n, n);
    if (!eigenvectors) {
        free_matrix(eigenvalues);
        return NULL;
    }

    const double EPSILON = 1e-10;
    const int MAX_ITER = 100;
    
    for (int k = 0; k < n; k++) {
        Complex lambda = eigenvalues->data[k][0];
        Matrix* shifted = create_matrix(n, n);
        Matrix* x = create_matrix(n, 1);
        
        // Initialize
        for (int i = 0; i < n; i++) {
            x->data[i][0] = (Complex){(double)rand()/RAND_MAX, 0};
            for (int j = 0; j < n; j++) {
                shifted->data[i][j] = m->data[i][j];
                if (i == j) {
                    shifted->data[i][j] = complex_subtract(shifted->data[i][j], lambda);
                }
            }
        }

        // Inverse iteration
        for (int iter = 0; iter < MAX_ITER; iter++) {
            Matrix* y = solve_linear_system(shifted, x);
            if (!y) {
                lambda.real += EPSILON;
                continue;
            }

            // Normalize
            double norm = 0;
            for (int i = 0; i < n; i++) {
                norm += y->data[i][0].real * y->data[i][0].real + 
                       y->data[i][0].imag * y->data[i][0].imag;
            }
            norm = sqrt(norm);

            if (norm < EPSILON) {
                free_matrix(y);
                continue;
            }

            // Check convergence
            double diff = 0;
            for (int i = 0; i < n; i++) {
                Complex normalized = {
                    y->data[i][0].real / norm,
                    y->data[i][0].imag / norm
                };
                Complex d = complex_subtract(normalized, x->data[i][0]);
                diff += sqrt(d.real * d.real + d.imag * d.imag);
                x->data[i][0] = normalized;
            }

            free_matrix(y);
            if (diff < EPSILON) break;
        }

        // Store eigenvector
        for (int i = 0; i < n; i++) {
            eigenvectors->data[i][k] = x->data[i][0];
        }

        free_matrix(x);
        free_matrix(shifted);
    }

    // Orthogonalize degenerate eigenvectors
    for (int i = 0; i < n - 1; i++) {
        Complex diff = complex_subtract(eigenvalues->data[i][0], 
                                      eigenvalues->data[i+1][0]);
        if (complex_abs(diff) < EPSILON) {
            Complex dot_product = {0, 0};
            for (int j = 0; j < n; j++) {
                dot_product = complex_add(dot_product, 
                    complex_multiply(eigenvectors->data[j][i], 
                                   complex_conjugate(eigenvectors->data[j][i+1])));
            }
            
            for (int j = 0; j < n; j++) {
                Complex proj = complex_multiply(dot_product, eigenvectors->data[j][i]);
                eigenvectors->data[j][i+1] = complex_subtract(
                    eigenvectors->data[j][i+1], proj);
            }
            
            // Normalize
            double norm = 0;
            for (int j = 0; j < n; j++) {
                norm += eigenvectors->data[j][i+1].real * eigenvectors->data[j][i+1].real +
                       eigenvectors->data[j][i+1].imag * eigenvectors->data[j][i+1].imag;
            }
            norm = sqrt(norm);
            
            if (norm > EPSILON) {
                for (int j = 0; j < n; j++) {
                    eigenvectors->data[j][i+1].real /= norm;
                    eigenvectors->data[j][i+1].imag /= norm;
                }
            }
        }
    }

    free_matrix(eigenvalues);
    return eigenvectors;
}

// Complex number operations
Complex complex_add(Complex a, Complex b) {
    Complex result = {
        .real = a.real + b.real,
        .imag = a.imag + b.imag
    };
    return result;
}

Complex complex_subtract(Complex a, Complex b) {
    Complex result = {
        .real = a.real - b.real,
        .imag = a.imag - b.imag
    };
    return result;
}

Complex complex_multiply(Complex a, Complex b) {
    Complex result = {
        .real = a.real * b.real - a.imag * b.imag,
        .imag = a.real * b.imag + a.imag * b.real
    };
    return result;
}

Complex complex_divide(Complex a, Complex b) {
    double denominator = b.real * b.real + b.imag * b.imag;
    Complex result = {
        .real = (a.real * b.real + a.imag * b.imag) / denominator,
        .imag = (a.imag * b.real - a.real * b.imag) / denominator
    };
    return result;
}

Complex complex_conjugate(Complex a) {
    Complex result = {
        .real = a.real,
        .imag = -a.imag
    };
    return result;
}

double complex_abs(Complex a) {
    return sqrt(a.real * a.real + a.imag * a.imag);
}

// Additional complex number operations
Complex complex_sqrt(Complex a) {
    double r = sqrt(complex_abs(a));
    double theta = atan2(a.imag, a.real) / 2.0;
    Complex result = {
        .real = r * cos(theta),
        .imag = r * sin(theta)
    };
    return result;
}

Complex complex_exp(Complex a) {
    double exp_real = exp(a.real);
    Complex result = {
        .real = exp_real * cos(a.imag),
        .imag = exp_real * sin(a.imag)
    };
    return result;
}

Complex complex_from_real(double real) {
    Complex result = {.real = real, .imag = 0};
    return result;
}

// Matrix conjugate transpose (Hermitian transpose)
Matrix* matrix_conjugate_transpose(Matrix* m) {
    if (!m) return NULL;
    
    Matrix* result = create_matrix(m->cols, m->rows);
    if (!result) return NULL;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j][i] = complex_conjugate(m->data[i][j]);
        }
    }
    return result;
}

// Check if matrix is Hermitian
int matrix_is_hermitian(Matrix* m) {
    if (!matrix_is_square(m)) return 0;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = i + 1; j < m->cols; j++) {
            Complex conj = complex_conjugate(m->data[i][j]);
            Complex diff = complex_subtract(conj, m->data[j][i]);
            if (complex_abs(diff) > 1e-10) return 0;
        }
    }
    return 1;
}

// Calculate matrix rank
int matrix_rank(Matrix* m) {
    if (!m) return 0;
    
    // Use QR decomposition to calculate rank
    Matrix* QR = matrix_qr_decomposition(m);
    if (!QR) return 0;
    
    int rank = 0;
    int min_dim = (m->rows < m->cols) ? m->rows : m->cols;
    
    // R is stored in the second block of QR
    for (int i = 0; i < min_dim; i++) {
        if (complex_abs(QR->data[m->rows + i][i]) > 1e-10) {
            rank++;
        }
    }
    
    free_matrix(QR);
    return rank;
}

// Solve linear system Ax = b using LU decomposition
Matrix* solve_linear_system(Matrix* A, Matrix* b) {
    if (!A || !b || !matrix_is_square(A) || A->rows != b->rows || b->cols != 1) 
        return NULL;
    
    int n = A->rows;
    Matrix* LU = matrix_lu_decomposition(A);
    if (!LU) return NULL;
    
    // Forward substitution (Ly = b)
    Matrix* y = create_matrix(n, 1);
    if (!y) {
        free_matrix(LU);
        return NULL;
    }
    
    for (int i = 0; i < n; i++) {
        y->data[i][0] = b->data[i][0];
        for (int j = 0; j < i; j++) {
            y->data[i][0] = complex_subtract(y->data[i][0], 
                complex_multiply(LU->data[i][j], y->data[j][0]));
        }
    }
    
    // Backward substitution (Ux = y)
    Matrix* x = create_matrix(n, 1);
    if (!x) {
        free_matrix(LU);
        free_matrix(y);
        return NULL;
    }
    
    for (int i = n - 1; i >= 0; i--) {
        x->data[i][0] = y->data[i][0];
        for (int j = i + 1; j < n; j++) {
            x->data[i][0] = complex_subtract(x->data[i][0], 
                complex_multiply(LU->data[i][j], x->data[j][0]));
        }
        x->data[i][0] = complex_divide(x->data[i][0], LU->data[i][i]);
    }
    
    free_matrix(LU);
    free_matrix(y);
    return x;
}

// Schur decomposition
Matrix* matrix_schur_decomposition(Matrix* m) {
    if (!matrix_is_square(m)) return NULL;
    
    int n = m->rows;
    Matrix* Q = create_matrix(n, n);
    Matrix* T = create_matrix(n, n);
    
    if (!Q || !T) {
        free_matrix(Q);
        free_matrix(T);
        return NULL;
    }
    
    // Initialize Q as identity and T as copy of m
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q->data[i][j] = (i == j) ? complex_from_real(1.0) : complex_from_real(0.0);
            T->data[i][j] = m->data[i][j];
        }
    }
    
    const int MAX_ITER = 100;
    const double EPSILON = 1e-10;
    
    // Iterative QR algorithm
    for (int iter = 0; iter < MAX_ITER; iter++) {
        Matrix* QR = matrix_qr_decomposition(T);
        if (!QR) {
            free_matrix(Q);
            free_matrix(T);
            return NULL;
        }
        
        // Extract Q1 and R1 from QR decomposition
        Matrix* Q1 = create_matrix(n, n);
        Matrix* R1 = create_matrix(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q1->data[i][j] = QR->data[i][j];
                R1->data[i][j] = QR->data[n + i][j];
            }
        }
        
        // Update Q and T
        Matrix* new_Q = matrix_multiply(Q, Q1);
        Matrix* new_T = matrix_multiply(R1, Q1);
        
        // Check convergence
        double diff = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                diff += complex_abs(new_T->data[i][j]);
            }
        }
        
        free_matrix(Q);
        free_matrix(T);
        Q = new_Q;
        T = new_T;
        
        free_matrix(QR);
        free_matrix(Q1);
        free_matrix(R1);
        
        if (diff < EPSILON) break;
    }
    
    // Combine Q and T into result matrix
    Matrix* result = create_matrix(2 * n, n);
    if (!result) {
        free_matrix(Q);
        free_matrix(T);
        return NULL;
    }
    
    // Copy Q and T into result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result->data[i][j] = Q->data[i][j];
            result->data[n + i][j] = T->data[i][j];
        }
    }
    
    free_matrix(Q);
    free_matrix(T);
    return result;
} 

// Matrix division (A/B = A * B^(-1))
Matrix* matrix_divide(Matrix* m1, Matrix* m2) {
    if (!m1 || !m2) return NULL;
    
    Matrix* inv = matrix_inverse(m2);
    if (!inv) return NULL;
    
    Matrix* result = matrix_multiply(m1, inv);
    free_matrix(inv);
    return result;
}

// Matrix power using repeated multiplication
Matrix* matrix_power(Matrix* m, Matrix* exp) {
    if (!m || !exp || !matrix_is_square(m) || exp->rows != 1 || exp->cols != 1) 
        return NULL;
    
    int power = (int)exp->data[0][0].real;
    if (power < 0) return NULL;
    
    if (power == 0) {
        // Return identity matrix
        Matrix* result = create_matrix(m->rows, m->rows);
        for (int i = 0; i < m->rows; i++) {
            result->data[i][i] = complex_from_real(1.0);
        }
        return result;
    }
    
    Matrix* result = matrix_multiply(m, m);
    for (int i = 2; i < power; i++) {
        Matrix* temp = matrix_multiply(result, m);
        free_matrix(result);
        result = temp;
    }
    return result;
}

int main(int argc, char *argv[]) {
    printf("\nComplex Matrix Calculator\n");
    printf("=======================\n");
    printf("Input Format:\n");
    printf("- Real matrix: [1, 2; 3, 4]\n");
    printf("- Complex matrix: [(1,0), (2,1); (0,1), (1,1)]\n");
    printf("- Operations: +, -, *, /, ^\n");
    printf("- Functions: det, transpose, inverse, eigenval, eigenvec, trace, rank, lu, qr\n");
    printf("- Example: [(1,1), (2,0); (0,1), (1,1)] * [(1,0), (1,1); (2,1), (0,1)]\n");
    printf("Enter 'quit' to exit\n");
    printf("=======================\n\n");
    printf("Enter expression: ");
    fflush(stdout);

    yyparse();
    return 0;
}
