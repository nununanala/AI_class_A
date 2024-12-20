import numpy as np

# Set seed for reproducibility
np.random.seed(5)

# Create matrices
matrix_2x3 = np.random.randint(1, 10, size=(2, 3))
matrix_3x4 = np.random.randint(1, 10, size=(3, 4))

print("Matrix 2x3:")
print(matrix_2x3)
print("\nMatrix 3x4:")
print(matrix_3x4)

# Multiply matrices using numpy
result_with_numpy = np.dot(matrix_2x3, matrix_3x4)
print("\nResult using numpy:")
print(result_with_numpy)

# Multiply matrices without using numpy
def matrix_multiply(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

result_without_numpy = matrix_multiply(matrix_2x3.tolist(), matrix_3x4.tolist())
print("\nResult without using numpy:")
print(result_without_numpy)