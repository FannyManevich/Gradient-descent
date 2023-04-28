import numpy as np
import math
import matplotlib.pyplot as plt
from decimal import *


# -- 1 --
def loadData(filename):
    data = np.loadtxt(filename)
    rows, fs = data.shape
    D = data[:, :fs-1]
    Y = data[:, fs-1]
    print("Read " + str(rows) + " rows")
    return D, Y

# -- 2 --
def selectSingle(D, n):
    #-- Selecting a single column --
    return D[:, n]


def addOnesColumn(D):
    new_d = D.reshape(-1, 1)

    #-- add a column of 1's --
    ones_column = np.ones((new_d.shape[0], 1), dtype=int)
    return np.hstack((ones_column, new_d))


# -- 3 --
def predictValue(rowD1, hypothesis):
    # -- Multiplication of two Matrices --
    return np.dot(rowD1, hypothesis)

# -- 4 --
def computeErrors(Data, Y, Hypothesis):
    if Data.shape[0] == Y.shape[0] :
        # -- Number of examples for study --
        m = Data.shape[0]
        # -- Vector of the prediction errors of the hypothesis --
        Errors = np.zeros(m)
        for i in range(m):
            Errors = np.insert(Errors, i, (predictValue(i, Hypothesis) - Y[i]))
    return Errors

# -- 5 --
def computeCost(Data, Y, Hypothesis):
    m = np.shape(Data)[0]
    error = computeErrors(Data, Y, Hypothesis)
    cost = 0
    for i in range(m):
        cost = cost + (error[i] ** 2)
        cost = cost / (2 * m)
    return cost

# -- 6 --
def computeGradient(Data, Errors):
    m = np.shape(Data)[0]
    n = np.shape(Data)[1]
    grad_vector = []
    vector = np.vectorize(np.int_)
    for i in range(m):
        for j in range(n):
            sum = ((Data[i][j]) * Errors[i])
        grad_vector.append(sum / m)
    return vector(grad_vector)

# -- 7 --
def updateHypothsis(Hypothesis, alpha, Gradient):
    return Hypothesis - alpha * Gradient

# -- 8 --
def gradientDescent(filename, alpha, max_iter, threshold):
    # -- Initialize --
    if 'alpha' not in locals():
        alpha = 0.1
    elif 'max_iter' not in locals():
        max_iter = 1000
    elif 'threshold' not in locals():
        threshold = 0.001
    cost_J = math.inf
    iter = 1
    Costs = np.empty(0)

    # -- 1 --
    # -- Loading data from file --
    D, Y = loadData("smartphone.txt")
    x1 = D[:, 0]
    x2 = D[:, 1]

    # -- Graph 1 --
    plt.scatter(x1, Y)
    plt.xlabel('x  Original price')
    plt.ylabel('y  New price')
    plt.title('Original price VS New price')
    plt.show()

    # -- Graph 2 --
    plt.scatter(x2, Y)
    plt.xlabel('x Age')
    plt.ylabel('y New price')
    plt.title('Age VS New price')
    plt.show()

    # -- 2 --
    # -- Adding a left unity column --
    vector = np.vectorize(np.int_)
    D = vector(addOnesColumn(selectSingle(D, 0)))

    Hypothesis = np.zeros(D.shape[0])

    while(iter < max_iter):

        # -- 4 --
        Data = vector(addOnesColumn(selectSingle(D, 0)))
        # -- Creating hypothesis vector of zeros
        Errors = (computeErrors(Data, Y, Hypothesis)[:, np.newaxis])

        iter = iter + 1

def main():


    # -- 3 --
    # rowD1 = D1[0, :]
    # hypothesis = [1, 1]
    # value = predictValue(rowD1, hypothesis)
    # print(value)




if __name__ == '__main__':
    main()

