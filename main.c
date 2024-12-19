#include "matrix.h"
#include "parser.tab.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int yyparse(void);

static MatrixError last_error = MATRIX_SUCCESS; // Last error code

// Error handling function
void matrix_perror(const char* op, MatrixError err) 
{
    last_error = err;
    switch(err) 
    {
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

// Get the last error code
MatrixError get_last_error(void) { return last_error; }

// Set the current error state
void set_last_error(MatrixError err) { last_error = err; }

// Create a new matrix
Matrix* create_matrix(int rows, int cols) 
{
    // Check for valid dimensions
    if (rows <= 0 || cols <= 0) 
    {
        matrix_perror("create_matrix", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (Element**)malloc(rows * sizeof(Element*));
    
    if (!m->data)  
    {
        // Memory allocation failed, free previously allocated memory
        free(m);
        return NULL;
    }
    
    for (int i = 0; i < rows; i++) 
    {
        m->data[i] = (Element*)calloc(cols, sizeof(Element));
        if (!m->data[i]) 
        {
            // Memory allocation failed, free previously allocated memory
            for (int j = 0; j < i; j++) 
            {
                free(m->data[j]);
            }
            free(m->data);
            free(m);
            return NULL;
        }
    }
    return m;
}

// Create a 1x1 matrix
Matrix* create_matrix_element(Element value) 
{
    Matrix* m = create_matrix(1, 1);
    if (m) m->data[0][0] = value;
    return m;
}

// Adds a new element to the matrix row
Matrix* add_element(Matrix* m, Element value) 
{    
    if (!m) return NULL; // Check for valid matrix
    
    Matrix* new_m = create_matrix(m->rows, m->cols + 1);
    if (!new_m) return NULL;
    
    // Copy existing elements
    for (int i = 0; i < m->rows; i++) 
    {
        for (int j = 0; j < m->cols; j++) 
        {
            new_m->data[i][j] = m->data[i][j];
        }
    }
    
    // Add new element
    new_m->data[0][m->cols] = value;
    free_matrix(m);
    return new_m;
}

// Appends a row to the matrix
Matrix* append_row(Matrix* m1, Matrix* m2) 
{    
    if (!m1 || !m2) // Check for valid matrices
    {
        matrix_perror("append_row", MATRIX_NULL_PTR);
        return NULL;
    }
    
    // Check for matching dimensions
    if (m1->cols != m2->cols) 
    {
        matrix_perror("append_row", MATRIX_DIM_MISMATCH);
        free_matrix(m1);
        free_matrix(m2);
        return NULL;
    }
    
    Matrix* new_m = create_matrix(m1->rows + 1, m1->cols);
    if (!new_m) return NULL; // Memory allocation failed
    
    // Copy existing rows
    for (int i = 0; i < m1->rows; i++) 
    {
        for (int j = 0; j < m1->cols; j++) 
        {
            new_m->data[i][j] = m1->data[i][j];
        }
    }
    
    // Add new row
    for (int j = 0; j < m2->cols; j++) 
    {
        new_m->data[m1->rows][j] = m2->data[0][j];
    }
    
    free_matrix(m1);
    free_matrix(m2);
    return new_m;
}

// Matrix addition
Matrix* matrix_add(Matrix* m1, Matrix* m2) 
{    
    if (!m1 || !m2) // Check for valid matrices
    {
        matrix_perror("addition", MATRIX_NULL_PTR);
        return NULL;
    }
    
    // Check for matching dimensions
    if (m1->rows != m2->rows || m1->cols != m2->cols) 
    {
        matrix_perror("addition", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) // Memory allocation failed
    {
        matrix_perror("addition", MATRIX_MEM_ERROR);
        return NULL;
    }
    
    // Perform addition
    for (int i = 0; i < m1->rows; i++) 
    {
        for (int j = 0; j < m1->cols; j++) 
        {
            result->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }
    return result;
}

// Matrix subtraction
Matrix* matrix_subtract(Matrix* m1, Matrix* m2)
{
    if (!m1 || !m2) // Check for valid matrices
    {
        matrix_perror("subtraction", MATRIX_NULL_PTR);
        return NULL;
    }
    
    // Check for matching dimensions
    if (m1->rows != m2->rows || m1->cols != m2->cols) 
    {
        matrix_perror("subtraction", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) // Memory allocation failed
    {
        matrix_perror("subtraction", MATRIX_MEM_ERROR);
        return NULL;
    }
    
    // Perform subtraction
    for (int i = 0; i < m1->rows; i++) 
    {
        for (int j = 0; j < m1->cols; j++) 
        {
            result->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }
    return result;
}

// Matrix multiplication
Matrix* matrix_multiply(Matrix* m1, Matrix* m2) 
{
    if (!m1 || !m2) // Check for valid matrices
    {
        matrix_perror("multiplication", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (m1->cols != m2->rows) // Check for matching dimensions
    {
        matrix_perror("multiplication", MATRIX_DIM_MISMATCH);
        return NULL;
    }
    
    Matrix* result = create_matrix(m1->rows, m2->cols);
    if (!result) // Memory allocation failed
    {
        matrix_perror("multiplication", MATRIX_MEM_ERROR);
        return NULL;
    }
    
    // Perform multiplication
    for (int i = 0; i < m1->rows; i++) 
    {
        for (int j = 0; j < m2->cols; j++) 
        {
            result->data[i][j] = 0;
            for (int k = 0; k < m1->cols; k++) 
            {
                result->data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }
    return result;
}

// Matrix transpose
Matrix* matrix_transpose(Matrix* m) 
{
    if (!m) // Check for valid matrix
    {
        matrix_perror("transpose", MATRIX_NULL_PTR);
        return NULL;
    }

    Matrix* result = create_matrix(m->cols, m->rows); // Create transposed matrix
    
    if (!result) 
    {
        matrix_perror("transpose", MATRIX_MEM_ERROR);
        return NULL;
    }
    
    // // Swap rows and columns
    for (int i = 0; i < m->rows; i++) 
    {
        for (int j = 0; j < m->cols; j++) 
        {
            result->data[j][i] = m->data[i][j];
        }
    }
    return result;
}

// Calculate determinant using recursive method
double calc_determinant(Matrix* m) 
{
    if (m->rows != m->cols) 
        return 0; // Check for square matrix
    if (m->rows == 1) 
        return m->data[0][0]; // Base case
    if (m->rows == 2) // Case for 2x2 matrix
    {
        return m->data[0][0] * m->data[1][1] - m->data[0][1] * m->data[1][0];
    }
    
    double det = 0; // Determinant value
    int sign = 1; // Sign multiplier
    
    for (int j = 0; j < m->cols; j++) 
    {
        Matrix* submatrix = create_matrix(m->rows - 1, m->cols - 1);
        if (!submatrix) 
            return 0;
        
        // Create submatrix
        for (int row = 1; row < m->rows; row++) 
        {
            int sub_col = 0;
            for (int col = 0; col < m->cols; col++) 
            {
                if (col == j) continue;
                submatrix->data[row-1][sub_col] = m->data[row][col];
                sub_col++;
            }
        }
        
        double sub_det = calc_determinant(submatrix); // Recursive call
        det += m->data[0][j] * sub_det * sign;
        sign = -sign; // Alternate sign
        free_matrix(submatrix); // Free submatrix memory
    }
    
    return det;
}

// Matrix determinant wrapper function
Matrix* matrix_determinant(Matrix* m) 
{
    if (!m) // Check for valid matrix
    {
        matrix_perror("determinant", MATRIX_NULL_PTR);
        return NULL;
    }
    if (!matrix_is_square(m)) // Check for square matrix
    {
        matrix_perror("determinant", MATRIX_NOT_SQUARE);
        return NULL;
    }
    double det = calc_determinant(m);
    return create_matrix_element(det);
}

// Free matrix memory
void free_matrix(Matrix* m) 
{
    if (!m) return; // Check for valid matrix
    
    // Free data array
    for (int i = 0; i < m->rows; i++) 
    {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Print matrix
void print_matrix(Matrix* m) 
{
    if (!m) return;
    
    int* col_widths = (int*)calloc(m->cols, sizeof(int));
    if (!col_widths) 
        return;
    
    // Find maximum width for each column
    for (int j = 0; j < m->cols; j++) 
    {
        for (int i = 0; i < m->rows; i++) 
        {
            double val = m->data[i][j];
            char buf[32];
            if (val - round(val) == 0) 
            {
                snprintf(buf, sizeof(buf), "%.0f", val); // Integer value
            } 
            else {
                // Floating point value with 2 decimal places
                snprintf(buf, sizeof(buf), "%.2f", val); 
            }
            int len = strlen(buf);
            if (len > col_widths[j]) 
                col_widths[j] = len;
        }
    }

    // Print matrix
    printf("[");
    for (int i = 0; i < m->rows; i++) 
    {
        if (i > 0) printf(" ");
        for (int j = 0; j < m->cols; j++) 
        {
            double val = m->data[i][j];
            char buf[32];
            if (val - round(val) == 0) 
            {
                snprintf(buf, sizeof(buf), "%.0f", val);
            } 
            else {
                snprintf(buf, sizeof(buf), "%.2f", val);
            }
            printf("%-*s", col_widths[j], buf);
            if (j < m->cols - 1) 
                printf("  ");
        }
        if (i < m->rows - 1) 
            printf("\n");
    }
    printf("]\n");
    free(col_widths);
}

// UCheck if matrix is square
int matrix_is_square(Matrix* m) 
{
    return m && (m->rows == m->cols);
}

// Calculate matrix trace
Matrix* matrix_trace(Matrix* m) 
{
    if (!m) 
    {
        matrix_perror("trace", MATRIX_NULL_PTR);
        return NULL;
    }
    if (!matrix_is_square(m)) 
    {
        matrix_perror("trace", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    double trace = 0;
    for (int i = 0; i < m->rows; i++) 
    {
        trace += m->data[i][i];
    }
    return create_matrix_element(trace);
}

// LU Decomposition using Doolittle's method
Matrix* matrix_lu_decomposition(Matrix* m) 
{
    if (!m)) // Check for valid matrix
    {
        matrix_perror("lu_decomposition", MATRIX_NULL_PTR);
        return NULL;
    }

    if(!matrix_is_square(m)) // Check for square matrix
    {
        matrix_perror("lu_decomposition", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    // Check if the matrix is decomposable
    for (int i = 0; i < m->rows-1; i++) 
    {
        // Check for zero diagonal element
        if (m->data[i][i] == 0) 
        {
            return NULL;
        }
    }
    
    int n = m->rows;
    Matrix* L = create_matrix(n, n);
    Matrix* U = create_matrix(n, n);
    
    // Initialize L's diagonal to 1
    for (int i = 0; i < n; i++) 
    {
        L->data[i][i] = 1.0;
    }
    
    // Calculate L and U
    for (int j = 0; j < n; j++) 
    {
        // Calculate U's j-th row
        for (int i = 0; i <= j; i++) 
        {
            double sum = 0;
            for (int k = 0; k < i; k++) 
            {
                sum += L->data[i][k] * U->data[k][j];
            }
            U->data[i][j] = m->data[i][j] - sum;
        }
        
        // Calculate L's j-th column
        for (int i = j + 1; i < n; i++) 
        {
            double sum = 0;
            for (int k = 0; k < j; k++) 
            {
                sum += L->data[i][k] * U->data[k][j];
            }
            L->data[i][j] = (m->data[i][j] - sum) / U->data[j][j];
        }
    }
    
    // Combine L and U into one matrix
    Matrix* result = create_matrix(2 * n, n);
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            result->data[i][j] = L->data[i][j];
            result->data[n + i][j] = U->data[i][j];
        }
    }
    
    free_matrix(L);
    free_matrix(U);
    return result;
}

// QR Decomposition using Gram-Schmidt process
Matrix* matrix_qr_decomposition(Matrix* m) 
{
    if (!m)
    {
        matrix_perror("qr_decomposition", MATRIX_NULL_PTR);
        return NULL;
    }
    
    int n = m->rows;
    int p = m->cols;
    Matrix* Q = create_matrix(n, n);
    Matrix* R = create_matrix(n, p);
    
    // Copy m to R
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < p; j++) 
        {
            R->data[i][j] = m->data[i][j];
        }
    }
    
    // Initialize Q to identity matrix
    for (int i = 0; i < n; i++) 
    {
        Q->data[i][i] = 1.0;
    }
    
    // Gram-Schmidt process
    for (int k = 0; k < p; k++) 
    {
        // Normalize k-th column
        double dot_product = 0;
        for (int i = 0; i < n; i++) 
        {
            dot_product += R->data[i][k] * R->data[i][k];
        }
        
        double norm = sqrt(dot_product);
        if (norm < 1e-10) continue;
        
        for (int i = 0; i < n; i++) 
        {
            R->data[i][k] /= norm;
            Q->data[i][k] = R->data[i][k];
        }
        
        // Orthogonalize remaining columns
        for (int j = k + 1; j < p; j++) 
        {
            double proj = 0;
            for (int i = 0; i < n; i++) 
            {
                proj += R->data[i][k] * R->data[i][j];
            }
            
            for (int i = 0; i < n; i++) 
            {
                R->data[i][j] -= R->data[i][k] * proj;
            }
        }
    }
    
    // Combine Q and R into result
    Matrix* result = create_matrix(2 * n, p);
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < p; j++) 
        {
            result->data[i][j] = Q->data[i][j];
            result->data[n + i][j] = R->data[i][j];
        }
    }
    
    free_matrix(Q);
    free_matrix(R);
    return result;
}

// Matrix inverse using Gauss-Jordan elimination
Matrix* matrix_inverse(Matrix* m) 
{
    if (!m) // Check for valid matrix
    {
        matrix_perror("inverse", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (!matrix_is_square(m)) // Check for square matrix
    {
        matrix_perror("inverse", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    // Check if matrix is invertible
    Matrix* det = matrix_determinant(m);
    if (!det) 
    {
        matrix_perror("inverse", MATRIX_SINGULAR);
        free_matrix(det);
        return NULL;
    }
    free_matrix(det);
    
    int n = m->rows;
    Matrix* augmented = create_matrix(n, 2 * n);
    if (!augmented) return NULL;

    // Create augmented matrix [A|I]
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            augmented->data[i][j] = m->data[i][j];
            augmented->data[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) 
    {
        // Find pivot
        double pivot = augmented->data[i][i];
        if (pivot == 0) 
        {
            matrix_perror("inverse", MATRIX_SINGULAR);
            free_matrix(augmented);
            return NULL;
        }

        // Normalize row i
        for (int j = 0; j < 2 * n; j++) 
        {
            augmented->data[i][j] /= pivot;
        }

        // Eliminate column i
        for (int j = 0; j < n; j++) 
        {
            if (i != j) 
            {
                double factor = augmented->data[j][i];
                for (int k = 0; k < 2 * n; k++) 
                {
                    augmented->data[j][k] -= factor * augmented->data[i][k];
                }
            }
        }
    }

    // Extract inverse matrix
    Matrix* inverse = create_matrix(n, n);
    if (!inverse) 
    {
        free_matrix(augmented);
        return NULL;
    }

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            inverse->data[i][j] = augmented->data[i][j + n];
        }
    }

    free_matrix(augmented);
    return inverse;
}

// Calculate eigenvalues using QR algorithm
Matrix* matrix_eigenvalues(Matrix* m) 
{
    if (!m) // Check for valid matrix
    {
        matrix_perror("eigenvalues", MATRIX_NULL_PTR);
        return NULL;
    }
    if (!matrix_is_square(m)) // Check for square matrix
    {
        matrix_perror("eigenvalues", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    // Copy matrix to A
    int n = m->rows;
    Matrix* A = create_matrix(n, n);
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            A->data[i][j] = m->data[i][j];
        }
    }
    
    // Initialize eigenvalues matrix
    Matrix* eigenvalues = create_matrix(n, 1);
    if (!eigenvalues) 
    {
        free_matrix(A);
        return NULL;
    }
    
    // QR Algorithm to calculate eigenvalues
    const int MAX_ITER = 100;
    for (int iter = 0; iter < MAX_ITER; iter++) 
    {
        // Calculate QR decomposition
        Matrix* QR = matrix_qr_decomposition(A);
        if (!QR) 
        {
            free_matrix(A);
            free_matrix(eigenvalues);
            return NULL;
        }
        
        // Extract Q and R from QR
        Matrix* Q = create_matrix(n, n);
        Matrix* R = create_matrix(n, n);
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                Q->data[i][j] = QR->data[i][j];
                R->data[i][j] = QR->data[n+i][j];
            }
        }
        
        // A = RQ
        Matrix* new_A = matrix_multiply(R, Q);
        free_matrix(Q);
        free_matrix(R);
        free_matrix(QR);
        
        if (!new_A) 
        {
            free_matrix(A);
            free_matrix(eigenvalues);
            return NULL;
        }
        
        // Check for convergence
        double diff = 0;
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < i; j++) 
            {
                diff += fabs(new_A->data[i][j]);
            }
        }
        
        free_matrix(A);
        A = new_A;
        
        if (diff < 1e-10) break; // Converged
    }
    
    // Extract eigenvalues from diagonal of A
    for (int i = 0; i < n; i++) 
    {
        eigenvalues->data[i][0] = A->data[i][i];
    }
    
    free_matrix(A);
    return eigenvalues;
}

// Calculate matrix rank
Matrix* matrix_rank(Matrix* m) 
{
    if (!m) // Check for valid matrix
    {
        matrix_perror("rank", MATRIX_NULL_PTR);
        return NULL;
    }
    
    // Use QR decomposition to calculate rank
    Matrix* QR = matrix_qr_decomposition(m);
    if (!QR) return 0;
    
    int rank = 0;
    int min_dim = (m->rows < m->cols) ? m->rows : m->cols;
    
    // R is stored in the second block of QR
    for (int i = 0; i < min_dim; i++) 
    {
        // Check if diagonal element is non-zero
        if (QR->data[m->rows + i][i]) 
        {
            rank++;
        }
    }
    
    free_matrix(QR);
    return create_matrix_element(rank);
    
}

// Matrix division (A/B = A * B^(-1))
Matrix* matrix_divide(Matrix* m1, Matrix* m2) 
{
    if (!m1 || !m2) // Check for valid matrices
    {
        matrix_perror("divide", MATRIX_NULL_PTR);
        return NULL;
    }

    Matrix* inv = matrix_inverse(m2);
    if (!inv) // Check if inverse exists
    {
        matrix_perror("divide", MATRIX_SINGULAR);
        return NULL;
    }
    
    Matrix* result = matrix_multiply(m1, inv);
    free_matrix(inv);
    return result;
}

// Matrix power using repeated multiplication
Matrix* matrix_power(Matrix* m, Matrix* exp) 
{
    if (!m || !exp) // Check for valid matrices
    {
        matrix_perror("power", MATRIX_NULL_PTR);
        return NULL;
    }
    
    if (!matrix_is_square(m)) // Check for square matrix
    {
        matrix_perror("power", MATRIX_NOT_SQUARE);
        return NULL;
    }
    
    // Check if exponent is a scalar
    if (exp->rows != 1 || exp->cols != 1) 
    {
        matrix_perror("power", MATRIX_INVALID_POWER);
        return NULL;
    }
    
    // Check if exponent is a non-negative integer
    int power = (int)exp->data[0][0];
    if (power < 0) 
    {
        matrix_perror("power", MATRIX_INVALID_POWER);
        return NULL;
    }
    
    if (power == 0) 
    {
        // Return identity matrix
        Matrix* result = create_matrix(m->rows, m->rows);
        for (int i = 0; i < m->rows; i++) 
        {
            result->data[i][i] = 1.0;
        }
        return result;
    }
    
    if (power == 1) 
    {
        // Return copy of original matrix
        Matrix* result = create_matrix(m->rows, m->rows);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) 
            {
                result->data[i][j] = m->data[i][j];
            }
        }
        return result;
    }
    
    // Recursive calculation of matrix power
    Matrix* result = matrix_multiply(m, m);
    for (int i = 2; i < power; i++) 
    {
        Matrix* temp = matrix_multiply(result, m);
        free_matrix(result);
        result = temp;
    }
    return result;
}

int main() 
{
    printf("\nMatrix Calculator\n");
    printf("================\n");
    printf("Input Format:\n");
    printf("- Matrix: [1, 2; 3, 4]\n");
    printf("- Operations: +, -, *, /, ^\n");
    printf("- Functions: det, transpose, inverse, trace, rank, eigenval, lu, qr\n");
    printf("- Example: [1,2;3,4] * [5,6;7,8]\n");
    printf("- Use parentheses to control operation order: ([1,2;3,4] + [5,6;7,8]) * [1,0;0,1]\n");
    printf("Enter 'help' for details, 'quit' to exit\n");
    printf("================\n");

    yyparse();
    return 0;
}
