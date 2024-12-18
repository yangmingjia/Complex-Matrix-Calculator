#include "matrix.h"
#include "parser.tab.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Forward declaration of the parser function */
int yyparse(void);

/* Global error state for matrix operations */
static MatrixError last_error = MATRIX_SUCCESS;

/* Error handling function implementation
 * Prints appropriate error message based on error type
 * @param op: Operation that caused the error
 * @param err: Type of error that occurred
 */
void matrix_perror(const char* op, MatrixError err) {
    last_error = err;
    switch(err) {
        case MATRIX_NULL_PTR:
            printf("Error: Invalid matrix operand\n");
            break;
        case MATRIX_DIM_MISMATCH:
            printf("Error: Matrix dimensions do not match\n");
            break;
        case MATRIX_NOT_SQUARE:
            printf("Error: Operation requires square matrix\n");
            break;
        case MATRIX_SINGULAR:
            printf("Error: Matrix is singular\n");
            break;
        case MATRIX_MEM_ERROR:
            printf("Error: Memory allocation failed\n");
            break;
        case MATRIX_INVALID_POWER:
            printf("Error: Invalid power operation\n");
            break;
        case MATRIX_CONVERGENCE_ERROR:
            printf("Error: Algorithm failed to converge\n");
            break;
        default:
            printf("Error: Unknown error\n");
    }
}

/* Get the last error that occurred
 * @return: Last error code
 */
MatrixError get_last_error(void) {
    return last_error;
}

/* Set the current error state
 * @param err: Error code to set
 */
void set_last_error(MatrixError err) {
    last_error = err;
}

/* Create a new matrix with specified dimensions
 * Allocates memory for the matrix and initializes it to zero
 * @param rows: Number of rows
 * @param cols: Number of columns
 * @return: Pointer to new matrix or NULL if allocation fails
 */
Matrix* create_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        matrix_perror("create_matrix", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (Element**)malloc(rows * sizeof(Element*));
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    for (int i = 0; i < rows; i++) {
        m->data[i] = (Element*)calloc(cols, sizeof(Element));
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

/* Create a 1x1 matrix with a given complex value
 * @param value: Complex value to store in matrix
 * @return: Pointer to new matrix or NULL if allocation fails
 */
Matrix* create_matrix_element(Element value) {
    Matrix* m = create_matrix(1, 1);
    if (m) m->data[0][0] = value;
    return m;
}

// Adds a new element to the matrix row
Matrix* add_element(Matrix* m, Element value) {
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
    if (!m1 || !m2) {
        matrix_perror("append_row", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (m1->cols != m2->cols) {
        matrix_perror("append_row", MATRIX_DIM_MISMATCH);
        free_matrix(m1);
        free_matrix(m2);
        return NULL;
    }
    
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
    if (!m1 || !m2) {
        matrix_perror("addition", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        matrix_perror("addition", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) {
        matrix_perror("addition", MATRIX_MEM_ERROR);
        return NULL;
    }
    
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }
    return result;
}

// Matrix subtraction
Matrix* matrix_subtract(Matrix* m1, Matrix* m2) {
    if (!m1 || !m2) {
        matrix_perror("subtraction", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        matrix_perror("subtraction", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) {
        matrix_perror("subtraction", MATRIX_MEM_ERROR);
        return NULL;
    }
    
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }
    return result;
}

// Matrix multiplication
Matrix* matrix_multiply(Matrix* m1, Matrix* m2) {
    if (!m1 || !m2) {
        matrix_perror("multiplication", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (m1->cols != m2->rows) {
        matrix_perror("multiplication", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* result = create_matrix(m1->rows, m2->cols);
    if (!result) {
        matrix_perror("multiplication", MATRIX_MEM_ERROR);
        return NULL;
    }
    
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            result->data[i][j] = 0;
            for (int k = 0; k < m1->cols; k++) {
                result->data[i][j] += m1->data[i][k] * m2->data[k][j];
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
double calc_determinant(Matrix* m) {
    if (m->rows != m->cols) return 0;
    if (m->rows == 1) return m->data[0][0];
    if (m->rows == 2) {
        return m->data[0][0] * m->data[1][1] - m->data[0][1] * m->data[1][0];
    }
    
    double det = 0;
    int sign = 1;
    
    for (int j = 0; j < m->cols; j++) {
        Matrix* submatrix = create_matrix(m->rows - 1, m->cols - 1);
        if (!submatrix) return 0;
        
        // Create submatrix
        for (int row = 1; row < m->rows; row++) {
            int sub_col = 0;
            for (int col = 0; col < m->cols; col++) {
                if (col == j) continue;
                submatrix->data[row-1][sub_col] = m->data[row][col];
                sub_col++;
            }
        }
        
        double sub_det = calc_determinant(submatrix);
        det += m->data[0][j] * sub_det * sign;
        sign = -sign;
        free_matrix(submatrix);
    }
    
    return det;
}

// Matrix determinant wrapper function
Matrix* matrix_determinant(Matrix* m) {
    if (!m || m->rows != m->cols) return NULL;
    double det = calc_determinant(m);
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

// Print matrix
void print_matrix(Matrix* m) {
    if (!m) return;
    
    // Calculate column widths
    int* col_widths = (int*)calloc(m->cols, sizeof(int));
    if (!col_widths) return;
    
    for (int j = 0; j < m->cols; j++) {
        for (int i = 0; i < m->rows; i++) {
            double val = m->data[i][j];
            char buf[32];
            if (fabs(val - round(val)) < 1e-10) {
                snprintf(buf, sizeof(buf), "%.0f", val);
            } else {
                snprintf(buf, sizeof(buf), "%.2f", val);
            }
            int len = strlen(buf);
            if (len > col_widths[j]) col_widths[j] = len;
        }
    }

    printf("[");
    for (int i = 0; i < m->rows; i++) {
        if (i > 0) printf(" ");
        for (int j = 0; j < m->cols; j++) {
            double val = m->data[i][j];
            char buf[32];
            if (fabs(val - round(val)) < 1e-10) {
                snprintf(buf, sizeof(buf), "%.0f", val);
            } else {
                snprintf(buf, sizeof(buf), "%.2f", val);
            }
            printf("%-*s", col_widths[j], buf);
            if (j < m->cols - 1) printf("  ");
        }
        if (i < m->rows - 1) printf("\n");
    }
    printf("]\n");
    free(col_widths);
}

// Utility function to check if matrix is square
int matrix_is_square(Matrix* m) {
    return m && (m->rows == m->cols);
}

// Calculate matrix trace (sum of diagonal elements)
double matrix_trace(Matrix* m) {
    if (!matrix_is_square(m)) return 0;
    
    double trace = 0;
    for (int i = 0; i < m->rows; i++) {
        trace += m->data[i][i];
    }
    return trace;
}

// LU Decomposition using Doolittle's method
Matrix* matrix_lu_decomposition(Matrix* m) {
    if (!m || !matrix_is_square(m)) return NULL;
    
    // 检查是否可分解（主元不为0）
    for (int i = 0; i < m->rows-1; i++) {
        if (fabs(m->data[i][i]) < 1e-10) {
            return NULL;
        }
    }
    
    int n = m->rows;
    Matrix* L = create_matrix(n, n);
    Matrix* U = create_matrix(n, n);
    
    // Initialize L's diagonal to 1
    for (int i = 0; i < n; i++) {
        L->data[i][i] = 1.0;
    }
    
    // Calculate L and U
    for (int j = 0; j < n; j++) {
        // Calculate U's j-th row
        for (int i = 0; i <= j; i++) {
            double sum = 0;
            for (int k = 0; k < i; k++) {
                sum += L->data[i][k] * U->data[k][j];
            }
            U->data[i][j] = m->data[i][j] - sum;
        }
        
        // Calculate L's j-th column
        for (int i = j + 1; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L->data[i][k] * U->data[k][j];
            }
            L->data[i][j] = (m->data[i][j] - sum) / U->data[j][j];
        }
    }
    
    // Combine L and U into one matrix
    Matrix* result = create_matrix(2 * n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result->data[i][j] = L->data[i][j];
            result->data[n + i][j] = U->data[i][j];
        }
    }
    
    free_matrix(L);
    free_matrix(U);
    return result;
}

// QR Decomposition using Gram-Schmidt process
Matrix* matrix_qr_decomposition(Matrix* m) {
    if (!m) return NULL;
    
    int n = m->rows;
    int p = m->cols;
    Matrix* Q = create_matrix(n, n);
    Matrix* R = create_matrix(n, p);
    
    // Copy m to R
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            R->data[i][j] = m->data[i][j];
        }
    }
    
    // Initialize Q to identity matrix
    for (int i = 0; i < n; i++) {
        Q->data[i][i] = 1.0;
    }
    
    // Gram-Schmidt process
    for (int k = 0; k < p; k++) {
        // Normalize k-th column
        double dot_product = 0;
        for (int i = 0; i < n; i++) {
            dot_product += R->data[i][k] * R->data[i][k];
        }
        
        double norm = sqrt(dot_product);
        if (norm < 1e-10) continue;
        
        for (int i = 0; i < n; i++) {
            R->data[i][k] /= norm;
            Q->data[i][k] = R->data[i][k];
        }
        
        // Orthogonalize remaining columns
        for (int j = k + 1; j < p; j++) {
            double proj = 0;
            for (int i = 0; i < n; i++) {
                proj += R->data[i][k] * R->data[i][j];
            }
            
            for (int i = 0; i < n; i++) {
                R->data[i][j] -= R->data[i][k] * proj;
            }
        }
    }
    
    // Combine Q and R into result
    Matrix* result = create_matrix(2 * n, p);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            result->data[i][j] = Q->data[i][j];
            result->data[n + i][j] = R->data[i][j];
        }
    }
    
    free_matrix(Q);
    free_matrix(R);
    return result;
}

// Matrix inverse using Gauss-Jordan elimination
Matrix* matrix_inverse(Matrix* m) {
    if (!m) {
        matrix_perror("inverse", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (!matrix_is_square(m)) {
        matrix_perror("inverse", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    // Check if matrix is invertible (determinant != 0)
    Matrix* det = matrix_determinant(m);
    if (!det || fabs(det->data[0][0]) < 1e-10) {
        matrix_perror("inverse", MATRIX_SINGULAR);
        free_matrix(det);
        return NULL;
    }
    free_matrix(det);
    
    int n = m->rows;
    Matrix* augmented = create_matrix(n, 2 * n);
    if (!augmented) return NULL;

    // Create augmented matrix [A|I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented->data[i][j] = m->data[i][j];
            augmented->data[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        double pivot = augmented->data[i][i];
        if (fabs(pivot) < 1e-10) {
            matrix_perror("inverse", MATRIX_SINGULAR);
            free_matrix(augmented);
            return NULL;
        }

        // Normalize row i
        for (int j = 0; j < 2 * n; j++) {
            augmented->data[i][j] /= pivot;
        }

        // Eliminate column i
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double factor = augmented->data[j][i];
                for (int k = 0; k < 2 * n; k++) {
                    augmented->data[j][k] -= factor * augmented->data[i][k];
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
    if (!m || !matrix_is_square(m)) return NULL;
    
    int n = m->rows;
    Matrix* A = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A->data[i][j] = m->data[i][j];
        }
    }
    
    Matrix* eigenvalues = create_matrix(n, 1);
    if (!eigenvalues) {
        free_matrix(A);
        return NULL;
    }
    
    // QR算法计算特征值
    const int MAX_ITER = 100;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 计算QR分解
        Matrix* QR = matrix_qr_decomposition(A);
        if (!QR) {
            free_matrix(A);
            free_matrix(eigenvalues);
            return NULL;
        }
        
        // 提取Q和R
        Matrix* Q = create_matrix(n, n);
        Matrix* R = create_matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q->data[i][j] = QR->data[i][j];
                R->data[i][j] = QR->data[n+i][j];
            }
        }
        
        // A = RQ
        Matrix* new_A = matrix_multiply(R, Q);
        free_matrix(Q);
        free_matrix(R);
        free_matrix(QR);
        
        if (!new_A) {
            free_matrix(A);
            free_matrix(eigenvalues);
            return NULL;
        }
        
        // 检查收敛性
        double diff = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                diff += fabs(new_A->data[i][j]);
            }
        }
        
        free_matrix(A);
        A = new_A;
        
        if (diff < 1e-10) break;
    }
    
    // 提取对角线元素作为特征值
    for (int i = 0; i < n; i++) {
        eigenvalues->data[i][0] = A->data[i][i];
    }
    
    free_matrix(A);
    return eigenvalues;
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
        if (fabs(QR->data[m->rows + i][i]) > 1e-10) {
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
            y->data[i][0] -= LU->data[i][j] * y->data[j][0];
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
            x->data[i][0] -= LU->data[i][j] * x->data[j][0];
        }
        x->data[i][0] /= LU->data[i][i];
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
            Q->data[i][j] = (i == j) ? 1.0 : 0.0;
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
                diff += fabs(new_T->data[i][j]);
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
    if (!m || !exp) {
        matrix_perror("power", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (!matrix_is_square(m)) {
        matrix_perror("power", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    if (exp->rows != 1 || exp->cols != 1) {
        matrix_perror("power", MATRIX_INVALID_POWER);
        return NULL;
    }
    
    int power = (int)exp->data[0][0];
    if (power < 0) {
        matrix_perror("power", MATRIX_INVALID_POWER);
        return NULL;
    }
    
    if (power == 0) {
        // Return identity matrix
        Matrix* result = create_matrix(m->rows, m->rows);
        for (int i = 0; i < m->rows; i++) {
            result->data[i][i] = 1.0;
        }
        return result;
    }
    
    if (power == 1) {
        // Return copy of original matrix
        Matrix* result = create_matrix(m->rows, m->rows);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                result->data[i][j] = m->data[i][j];
            }
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

// 添加Frobenius范数计算
double matrix_norm(Matrix* m) {
    if (!m) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            sum += m->data[i][j] * m->data[i][j];
        }
    }
    return sqrt(sum);
}

// 矩阵指数函数 (e^A)
Matrix* matrix_exp(Matrix* m) {
    if (!matrix_is_square(m)) {
        matrix_perror("exp", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    int n = m->rows;
    Matrix* result = create_matrix(n, n);
    if (!result) return NULL;
    
    // 初始化单位矩阵作为第一项
    Matrix* sum = create_matrix(n, n);
    Matrix* term = create_matrix(n, n);
    if (!sum || !term) {
        free_matrix(sum);
        free_matrix(term);
        free_matrix(result);
        return NULL;
    }
    
    // 初始化为单位矩阵 I
    for (int i = 0; i < n; i++) {
        sum->data[i][i] = 1.0;  // sum = I
        term->data[i][i] = 1.0; // term = I
    }
    
    double factorial = 1.0;
    
    // 使用泰勒级数展开计算
    // exp(A) = I + A + A^2/2! + A^3/3! + ...
    for (int k = 1; k <= 20; k++) {  // 20阶泰展开
        factorial *= k;
        Matrix* new_term = matrix_multiply(term, m);
        if (!new_term) break;
        
        free_matrix(term);
        term = new_term;
        
        // 加上新项 term/k!
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum->data[i][j] += term->data[i][j] / factorial;
            }
        }
    }
    
    free_matrix(term);
    result = sum;
    return result;
}

Matrix* matrix_log(Matrix* m) {
    if (!matrix_is_square(m)) {
        matrix_perror("log", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    // 检查特征值是否都为正
    Matrix* eigenvals = matrix_eigenvalues(m);
    if (!eigenvals) return NULL;
    
    for (int i = 0; i < m->rows; i++) {
        if (eigenvals->data[i][0] <= 0) {
            matrix_perror("log", MATRIX_SINGULAR);
            free_matrix(eigenvals);
            return NULL;
        }
    }
    free_matrix(eigenvals);
    
    // 使用Schur分解计算矩阵对数
    Matrix* schur = matrix_schur_decomposition(m);
    if (!schur) return NULL;
    
    int n = m->rows;
    Matrix* Q = create_matrix(n, n);
    Matrix* T = create_matrix(n, n);
    
    // 提取Q和T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q->data[i][j] = schur->data[i][j];
            T->data[i][j] = schur->data[n+i][j];
        }
    }
    
    // 计算对角矩阵的对数
    Matrix* logT = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        logT->data[i][i] = log(T->data[i][i]);
    }
    
    // 计算最终结果：Q * log(T) * Q^T
    Matrix* temp = matrix_multiply(logT, matrix_transpose(Q));
    Matrix* result = matrix_multiply(Q, temp);
    
    free_matrix(schur);
    free_matrix(Q);
    free_matrix(T);
    free_matrix(logT);
    free_matrix(temp);
    
    return result;
}

// Eigenvector calculation
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
        double lambda = eigenvalues->data[k][0];
        Matrix* shifted = create_matrix(n, n);
        Matrix* x = create_matrix(n, 1);
        
        // Initialize
        for (int i = 0; i < n; i++) {
            x->data[i][0] = (double)rand()/RAND_MAX;
            for (int j = 0; j < n; j++) {
                shifted->data[i][j] = m->data[i][j];
                if (i == j) {
                    shifted->data[i][j] = shifted->data[i][j] - lambda;
                }
            }
        }

        // Inverse iteration
        for (int iter = 0; iter < MAX_ITER; iter++) {
            Matrix* y = solve_linear_system(shifted, x);
            if (!y) {
                lambda += EPSILON;
                continue;
            }

            // Normalize
            double norm = 0;
            for (int i = 0; i < n; i++) {
                norm += y->data[i][0] * y->data[i][0];
            }
            norm = sqrt(norm);

            if (norm < EPSILON) {
                free_matrix(y);
                continue;
            }

            // Check convergence
            double diff = 0;
            for (int i = 0; i < n; i++) {
                double normalized = y->data[i][0] / norm;
                double d = normalized - x->data[i][0];
                diff += d * d;
                x->data[i][0] = normalized;
            }

            free_matrix(y);
            if (sqrt(diff) < EPSILON) break;
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
        double diff = eigenvalues->data[i][0] - eigenvalues->data[i+1][0];
        if (fabs(diff) < EPSILON) {
            // Gram-Schmidt orthogonalization
            double dot_product = 0;
            for (int j = 0; j < n; j++) {
                dot_product += eigenvectors->data[j][i] * eigenvectors->data[j][i+1];
            }
            
            for (int j = 0; j < n; j++) {
                eigenvectors->data[j][i+1] -= dot_product * eigenvectors->data[j][i];
            }
            
            // Normalize
            double norm = 0;
            for (int j = 0; j < n; j++) {
                norm += eigenvectors->data[j][i+1] * eigenvectors->data[j][i+1];
            }
            norm = sqrt(norm);
            
            if (norm > EPSILON) {
                for (int j = 0; j < n; j++) {
                    eigenvectors->data[j][i+1] /= norm;
                }
            }
        }
    }

    free_matrix(eigenvalues);
    return eigenvectors;
}

int main(int argc, char *argv[]) {
    printf("\nMatrix Calculator\n");
    printf("================\n");
    printf("Input Format:\n");
    printf("- Matrix: [1, 2; 3, 4]\n");
    printf("- Operations: +, -, *, /, ^\n");
    printf("- Functions: det, transpose, inverse, eigenval, eigenvec, trace, rank, lu, qr, exp, log, norm\n");
    printf("- Example: [1,2;3,4] * [5,6;7,8]\n");
    printf("- Use parentheses to control operation order: ([1,2;3,4] + [5,6;7,8]) * [1,0;0,1]\n");
    printf("Enter 'help' for details, 'quit' to exit\n");
    printf("================\n\n");

    printf("Enter expression: ");
    yyparse();
    return 0;
}
