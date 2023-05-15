from main import addOnesColumn
from main import selectSingle
import numpy as np
import math


# Fany Manevich 206116725
# Ilona Grand 316179548

# -- 1.1 --
# -- calculating sigmoid g(z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -- 1.2 --
def predictValue(Example, Hypothesis):
    # -- Multiplication of two Matrices --
    h = np.dot(Example, Hypothesis)

    return sigmoid(h)


# # -- 2.1 --
def computeCostAndGradient(D, Y, Hypothesis):
    J = 0
    m = Y.shape[0]
    Gradient = np.zeros(D.shape[0])

    for i in range(m):
        # prediction value
        h = predictValue(D[i], Hypothesis)

        # calculating the price
        J += (-Y[i] * math.log(h) - (1 - Y[i]) * math.log(1 - h))

        # calculating the error i
        error = h - Y[i]

        # updating gradient matrix
        for j in range(D[0].shape):
            Gradient[j] += error * D[i][j]

        J /= m
        Gradient /= m

    return J, Gradient
# # -- 2.2 --
#
# # -- 4 --
# def updateHypothsis(Hypothesis, alpha, Gradient):
#
#     return newHypo
# # -- 5 --
# def gradientDescent(Data, Y, Hypothesis, alpha, max_iter, threshold):

def main():
    # -- 0 --
    # -- Loading data from file --
    data = np.genfromtxt('ex2data1.txt', dtype=float, delimiter=',')
    fs = data.size
    D = data[:, :fs - 1]
    x1 = D[:, 0]
    x2 = D[:, 1]
    Y = D[:, 2]

    # -- Adding a left unity column --
    vector = np.vectorize(np.float_)
    D = vector(addOnesColumn(selectSingle(D, 2)))

    # -- 1 --
    # -- checking --
    # matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
    # matrix2 = np.array([[-1, -2, -3], [-4, -5, -6]])

    print(predictValue(Y, x1))

    # -- 2.1 --
    Hypothesis = [-10, 0.8, 0.08]
    print(computeCostAndGradient(D, Y, Hypothesis))

if __name__ == '__main__':
    main()
