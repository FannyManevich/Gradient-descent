import numpy as np
import math


# Fany Manevich 206116725
# Ilona Grand 316179548

# -- 0 --
# -- Loading data from file --
def loadData(filename):
    data = np.genfromtxt('ex2data1.txt', dtype=float, delimiter=',')
    fs = data.shape[1]
    D = data[:, [0, 1]]
    Y = data[:, fs - 1]
    return D, Y
def addOnesColumn(D):
    new_d = D.reshape(-1, 1)

    # -- add a column of 1's --
    ones_column = np.ones((new_d, 1), dtype=float)
    return np.concatenate((ones_column, D), axis=1)

# -- 1.1 --
# -- calculating sigmoid g(z)
def sigmoid(z):
    # z is scalar
    if z.ndim == 0:
        return 1 / (1 + np.exp(-z))

    # z is vector
    elif z.ndim == 1:
            # creating new vector
            newZ = np.empty(z.size)

            # applying the sigmoid function on each cell
            for i in range(z.size):
                sig = 1 / (1 + np.exp(-z[i]))
                newZ[i] = sig
            return newZ
    # z is matrix
    else:
        # creating new matrix
        newZ = np.empty(z.shape)

        # applying the sigmoid function on each cell
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                sig = 1 / (1 + np.exp(-z[i][j]))
                newZ[i][j] = sig
        return newZ



# -- 1.2 --
def predictValue(Example, Hypothesis):
    # -- Multiplication of two Matrices --
    h = Example * Hypothesis
    return sigmoid(h)


# # -- 2.1 --
def computeCostAndGradient(D, Y, Hypothesis):
    J = 0
    m = Y.shape[0]
    Gradient = np.zeros(D.shape[0])

    for i in range(m):
        # prediction value
        h = predictValue(D[i], Hypothesis)

        if h == 0 or (1 - h) == 0:
            h = 0.0001

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
def updateHypothsis(Hypothesis, alpha, Gradient):
    newHypo = Hypothesis - alpha * Gradient
    return newHypo
# # -- 5 --
# def gradientDescent(Data, Y, Hypothesis, alpha, max_iter, threshold):

def main():
    # -- 0 --
    # -- Loading data from file --
    D, Y = loadData('ex2data1.txt')
    # -- Adding a left unity column --
    vector = np.vectorize(np.float_)
    D = vector(addOnesColumn(D))
    print(D)

    # -- 1 --
    # -- checking --
    # matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
    # matrix2 = np.array([[-1, -2, -3], [-4, -5, -6]])

    Hypothesis = np.zeros(D.shape)
    print(predictValue(D, Hypothesis))

    # -- 2.1 --
    Hypothesis = [-10, 0.8, 0.08]

    # print(computeCostAndGradient(D, Y, Hypothesis))

if __name__ == '__main__':
    main()
