import numpy as np


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


# -- add a column of 1's --
def addOnesColumn(D):
    return np.insert(D, 0, values=1, axis=1)


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
        for i in range(z.size - 1):
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
    n = D.shape[1]
    m = Y.shape[0]
    print("n = ", n)
    vector = np.vectorize(np.float_)
    error = np.array([])
    Gradient = np.zeros_like(Hypothesis)

    for i in range(0 ,m):
        # prediction value
        h = predictValue(D[i], Hypothesis)

        h = np.where(h == 0, 0.0001, h)
        h = np.where(h == 1, 0.9999, h)
        sum = 0
        for j in range(n - 1):
            # calculating the price
            J += (-Y[i] * np.log(h[j]) - (1 - Y[i]) * np.log(1 - h[j]))

            # calculating the error i
            error = np.insert(error, j, h - Y[i])

            # calculating sum for gradient
            Gradient[j] += error[j] * D[i][j]
        # updating gradient matrix
        Gradient /= m
        # print(Gradient)
    np.set_printoptions(suppress=True)
    print(Gradient)
    J /= m
    return vector(Gradient), J

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

    # -- 1 --
    # -- checking --
    # matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
    # matrix2 = np.array([[-1, -2, -3], [-4, -5, -6]])

    Hypothesis = np.zeros(D.shape)
    # print(predictValue(D, Hypothesis))

    # -- 2.1 --
    Hypothesis = np.array([0.08, 0.8, -10])
    Gradient, J = computeCostAndGradient(D, Y, Hypothesis)
    np.set_printoptions(suppress=True)
    print(Gradient)
    np.set_printoptions(suppress=True)
    print(J)


if __name__ == '__main__':
    main()
