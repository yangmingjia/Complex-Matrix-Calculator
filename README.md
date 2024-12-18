# Complex Matrix Calculator

A powerful calculator supporting both real and complex matrix operations.

## Table of Contents

- [Features](#features)
- [Basic Usage](#basic-usage)
  - [Matrix Input Format](#matrix-input-format)
  - [Basic Operations](#basic-operations)
- [Detailed Function Reference](#detailed-function-reference)
  - [Matrix Functions](#matrix-functions)
  - [Advanced Operations](#advanced-operations)
- [Complex Numbers](#complex-numbers)
  - [Complex Operations](#complex-operations)
- [Advanced Usage](#advanced-usage)
  - [Matrix Chaining](#matrix-chaining)
  - [Matrix Properties](#matrix-properties)
  - [Special Matrices](#special-matrices)
- [Error Handling](#error-handling)
- [Tips and Tricks](#tips-and-tricks)
- [Performance Considerations](#performance-considerations)

## Features

- Complex number support
- Basic matrix operations (+, -, *, /, ^)
- Advanced matrix operations (determinant, inverse, eigenvalues, etc.)
- Matrix decompositions (LU, QR, Schur)
- Real-time interactive mode
- Automatic format detection (real/complex)

## Basic Usage

### Matrix Input Format

1. Real Matrix:
```
Input: [1, 2; 3, 4]
Output:
[
  1  2
  3  4
]
```

2. Complex Matrix:
```
Input: [(1,1), (2,0); (0,1), (1,1)]
Output:
[
  (1, 1)  (2, 0)
  (0, 1)  (1, 1)
]
```

### Basic Operations

1. Addition:
```
Input: [1, 2; 3, 4] + [5, 6; 7, 8]
Output:
[
  6  8
  10  12
]
```

2. Multiplication:
```
Input: [1, 2; 3, 4] * [5, 6; 7, 8]
Output:
[
  19  22
  43  50
]
```

## Detailed Function Reference

### Matrix Functions

1. Determinant:
```
Input: det [1, 2; 3, 4]
Output:
[
  -2
]
```

2. Transpose:
```
Input: transpose [1, 2, 3; 4, 5, 6]
Output:
[
  1  4
  2  5
  3  6
]
```

3. Matrix Inverse:
```
Input: inverse [1, 2; 3, 4]
Output:
[
  -2   1
  1.5  -0.5
]
```

### Advanced Operations

1. Eigenvalues:
```
Input: eigenval [(1,1), (2,0); (0,1), (1,1)]
Output:
[
  (2.41, 1)
  (-0.41, 1)
]
```

2. LU Decomposition:
```
Input: lu [1, 2; 3, 4]
Output:
[
  1    0
  3    1
  --------
  1    2
  0   -2
]
```

## Complex Numbers

### Complex Operations

1. Complex Addition:
```
Input: [(1,1), (0,0)] + [(0,1), (1,1)]
Output:
[
  (1, 2)  (1, 1)
]
```

2. Complex Multiplication:
```
Input: [(1,1), (2,0)] * [(1,0), (0,1)]
Output:
[
  (1, 1)  (0, 1)
]
```

## Advanced Usage

### Matrix Chaining

1. Multiple Operations:
```
Input: det (transpose [1, 2; 3, 4] * [5, 6; 7, 8])
Output:
[
  -4
]
```

2. Complex Expressions:
```
Input: eigenval ([(1,1), (2,0); (0,1), (1,1)] + [(1,0), (0,0); (0,0), (1,0)])
Output:
[
  (3.41, 1)
  (0.59, 1)
]
```

### Matrix Properties

1. Rank:
```
Input: rank [1, 2, 3; 4, 5, 6; 7, 8, 9]
Output:
[
  2
]
```

2. Trace:
```
Input: trace [(1,1), (2,0); (0,1), (1,1)]
Output:
[
  (2, 2)
]
```

### Special Matrices

1. Identity Matrix Power:
```
Input: [1, 0; 0, 1] ^ 3
Output:
[
  1  0
  0  1
]
```

2. Complex Rotation:
```
Input: [(0,1), (-1,0); (1,0), (0,1)] * [(1,0), (2,0)]
Output:
[
  (-2, 1)
  (1, 2)
]
```

## Error Handling

1. Dimension Mismatch:
```
Input: [1, 2] + [1, 2, 3]
Output: Error: Matrix dimensions do not match for addition
```

2. Singular Matrix:
```
Input: inverse [1, 1; 1, 1]
Output: Error: Matrix is singular (non-invertible)
```

## Tips and Tricks

1. Use parentheses to control operation order:
```
Input: ([1, 2; 3, 4] + [5, 6; 7, 8]) * [1, 0; 0, 1]
```

2. Complex numbers can be entered in polar form:
```
Input: [(1,1), (1,-1)]  // represents 1+i, 1-i
```

3. For large matrices, use semicolons to separate rows:
```
Input: [1, 2, 3; 4, 5, 6; 7, 8, 9]
```

## Performance Considerations

- Optimal matrix size: < 100x100
- Memory usage: O(n²) for n×n matrix
- Operation complexity:
  - Addition/Subtraction: O(n²)
  - Multiplication: O(n³)
  - Determinant/Inverse: O(n³)
  - Eigenvalues: O(n³)
