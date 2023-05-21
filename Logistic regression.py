import numpy as np
import matplotlib.pyplot as plt
from plotDecisionBoundary import plotDecisionBoundary


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
    h = np.dot(Example, Hypothesis)
    return sigmoid(h)


# -- 2 --
def computeRegularizedCostAndGradient(D, Y, Hypothesis, Lambda):
    J = 0
    regularization = 0
    n = Hypothesis.shape[0]
    m = Y.shape[0]

    vector = np.vectorize(np.float_)
    errors = np.array([])
    Gradient = np.array([])
    h = np.array([])

    for i in range(m):
        # prediction value
        h = predictValue(D[i], Hypothesis)

        # calculating the error i
        errors = np.append(errors, (h - Y[i]))

        h = np.where(h == 0, 0.0001, h)
        h = np.where(h > 0.999, 0.999, h)

        J += (-Y[i] * np.log(h) - (1 - Y[i]) * np.log(1 - h))

    J /= m
    np.set_printoptions(suppress=True)
    # new_errors = np.reshape(errors, (100, 3))

    for j in range(n):
        # regularization term to cost
        regularization = (Lambda / (2 * m)) * np.sum(Hypothesis[1:] ** 2)

    J += regularization
    # updating gradient matrix
    errorT = np.transpose(errors)
    Gradient = np.dot(errorT, D)
    Gradient /= m
    gr = Gradient.shape[0]

    for g in range(gr):
        Gradient[g] += (Lambda / m) * Hypothesis[g]

    return vector(Gradient), J


# -- 4 --
def updateHypothsis(Hypothesis, alpha, Gradient):
    newHypo = Hypothesis - alpha * Gradient
    return newHypo


# # -- 5 --
def gradientDescent(Data, Y, Hypothesis, alpha, max_iter, threshold):
    # -- Initialize --
    cost_j = float('inf')
    cost: list = []
    costs = np.array([])
    theta = np.array([-8, 2, -0.5])

    # ---- before computing regularization cost & gradient ---
    plotDecisionBoundary(theta, Data, Y)

    iter = 1
    Lambda = 0
    costs.append(cost_j)
    Gradient, cost_j = computeRegularizedCostAndGradient(Data, Y, Hypothesis, Lambda)
    Hypothesis = updateHypothsis(Hypothesis, alpha, Gradient)
    iter = iter + 1

    while iter < max_iter + 1 and (iter <= 1 or np.abs(costs[iter - 2] - costs[iter - 1]) > threshold):
        costs.append(cost_j)
        Gradient, cost_j = computeRegularizedCostAndGradient(Data, Y, Hypothesis, Lambda)
        Hypothesis = updateHypothsis(Hypothesis, alpha, Gradient)
        print(Hypothesis)
        iter = iter + 1

        if abs(costs[iter - 1] - costs[iter]) < threshold:
            print(
                f"Breaking loop because improvement is under threshold. {costs[iter - 1] - costs[iter]} < {threshold}")
            break

    print("Gradient descent terminating after {} iterations. Improvement was :  {} -below threshold  {}".format(iter,
                                                                                                                abs(
                                                                                                                    costs[
                                                                                                                        iter - 1] -
                                                                                                                    costs[
                                                                                                                        iter]),
                                                                                                                threshold))
    costs = np.delete(costs, [0])

    x = np.arange(max_iter)
    plt.scatter(x, costs)
    plt.xlabel("x - Iteration")
    plt.ylabel("y - Cost")
    plt.title('Iteration VS Cost')
    plt.show()

    # ---- after computing regularization cost & gradient ---
    plotDecisionBoundary(Hypothesis, Data, Y)

    return Hypothesis, costs


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

    # Hypothesis = np.zeros(D.shape)
    # print(predictValue(D, Hypothesis))

    # -- 2 --
    Hypothesis = np.array([-10, 0.8, 0.08])
    Lambda = 0
    Gradient, J = computeRegularizedCostAndGradient(D, Y, Hypothesis, Lambda)

    np.set_printoptions(suppress=True)
    print(Gradient)
    np.set_printoptions(suppress=True)
    print(J)

    # -- 5 --
    alpha = 0.001
    max_iter = 1000
    threshold = 0.0001
    gradientDescent(D, Y, Hypothesis, alpha, max_iter, threshold)

if __name__ == '__main__':
    main()
