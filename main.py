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
    # new_d = D.reshape(-1, 1)

    #-- add a column of 1's --
    ones_column = np.ones((D.shape[0], 1), dtype=int)
    return np.hstack((ones_column, D))


# -- 3 --
def predictValue(rowD1, hypothesis):
    # -- Multiplication of two Matrices --
    return np.dot(rowD1, hypothesis)


# -- 4 --
def computeErrors(Data, Y, Hypothesis):

    # -- Number of examples for study --
    m = Data.shape[0]
    errors = np.array([])
    
    if Data.shape[0] == Y.shape[0]:
        # -- Vector of the prediction errors of the hypothesis --
        for i in range(m):
            errors = np.insert(errors, i, (predictValue(Data[i, :], Hypothesis) - Y[i]))
    return errors


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
    vector = np.vectorize(np.float_)
    for j in range(n):
        sum = 0 
        for i in range(m):
            sum += ((Data[i][j]) * Errors[i])
        grad_vector.append(sum / m)
    return vector(grad_vector)


# -- 7 --
def updateHypothsis(Hypothesis, alpha, Gradient):
    return Hypothesis - alpha * Gradient


# -- 8 --
def gradientDescent(filename, alpha, max_iter, threshold):
    cost_j = float('inf')
    cost: list = []

    # -- Initialize --
    if 'alpha' not in locals():
        alpha = 0.1
    elif 'max_iter' not in locals():
        max_iter = 1000
    elif 'threshold' not in locals():
        threshold = 0.001

    cost_j = math.inf
    cost.append(cost_j)
    iter = 0
    costs = np.array([])
    costs = np.append(costs, np.inf)

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
    D = vector(addOnesColumn(D))

    Hypothesis = np.zeros(3)

    while iter < max_iter:
        print(iter)
        # -- 4 --
        Data = D
        # -- Creating hypothesis vector of zeros
        # Errors = (computeErrors(Data, Y, Hypothesis)[:, np.newaxis])
        Errors = (computeErrors(Data, Y, Hypothesis))
        # -- 5 --
        costs = np.append(costs, computeCost(Data, Y, Hypothesis))
        # -- 6 --
        Gradient = computeGradient(Data, Errors)
        # -- 7 --
        Hypothesis = updateHypothsis(Hypothesis, alpha, Gradient)
        if abs(costs[iter-1] - costs[iter]) < threshold:
            print(f"Breaking loop because improvement is under threshold. {costs[iter-1] - costs[iter]} < {threshold}")
            break
        iter = iter + 1

    print("Gradient descent terminating after {} iterations. Improvement was :  {} -below threshold  {}".format(iter, abs(costs[iter-1] - costs[iter]), threshold))
    costs = np.delete(costs, [0])
    # Hypothesis, cost, x = gradientDescent(
    #     'smartphone.txt', 0.05, 2000, 0.0001)  # calculate gradient descent

    x = np.arange(max_iter)
    print(f"size of x: {np.shape(x)}")
    print(f"size of x: {np.shape(costs)}")
    plt.scatter(x, costs)
    plt.xlabel("x - Iteration")
    plt.ylabel("y - Cost")
    plt.title('Iteration VS Cost')
    plt.show()

    return Hypothesis, costs


def main():
    gradientDescent("smartphone", 0.1, 1000, 0.001)


if __name__ == '__main__':
    main()
