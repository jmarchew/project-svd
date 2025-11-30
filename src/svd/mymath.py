#This file includes definitions of basic mathematical functions required for SVD

import math

#############################
#########  VECTORS  #########
#############################

#function computing dot scalar product of 2 vectors
def vec_dot(a,b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

#Norm L2 of vector

def norm(x):
    return math.sqrt(dot(x,x))

#Normalization of vecotr

def normalize(x):
    n = norm(x)
    if n == 0:
        raise ValueError("0 vector normalization is forbidden!")
    result = []
    for xi in x:
        result.append(xi/n)
    return result

#Multiplication of vector x by scalar a

def vector_scalar_mul(a, x):
    result = []
    for xi in x:
        result.append(xi*a)
    return result

#Adding 2 vectors
def vec_add(x, y):
    result = []
    for i in range(len(x)):
        result.append(x[i] + y[i])
    return result

#############################
#########  MATRICES  ########
#############################

#Matrix transposition

def mat_transpose(A):
    rows = len(A)
    cols = len(A[0])

    T = []

    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(A[i][j])
        T.append(new_row)
    return T

#Multiplication of matrix by vector

def mat_vector_mul(A, x):
    result = []
    for row in A:
        result.append(vec_dot(row, x)):
    return result

#Multiplication of 2 matrices

def matrix_mul(A, B):
    BT = mat_transpose(B)

    result = []
    for row in A:
        new_row = []
        for col in BT:
            new_row.append(vec_dot(row, col))
        result.append(new_row)
    return result

#outer multiplication of two vectors resulting in matrix
def outer_mul(x, y):
    result = []
    for i in range(len(x)):
        row = []
        for j in range(len(y)):
            row.append(x[i] *y[j])
        result.append(row)
    return result

#substraction of matrices
def mat_sub(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] - B[i][j])
        result.append(row)
    return result

#convergence - checking if vectors are close

def converged(x_old, x_new, eps=1e-9):
    diff = mat_sub(x_old, x_new)
    return norm(diff) < eps