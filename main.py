import numpy as np
import matplotlib.pyplot as plt
from decimal import *


# -- 1 --
def loadData(filename):
    data = np.loadtxt(filename)
    rows, fs = data.shape
    D = data[:, :fs-1]
    Y = data[:, fs-1]
    # print("Read " + str(rows) + " rows")
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
    #-- Multiplication of two Matrices --
    return np.dot(rowD1, hypothesis)

def main():
    # -- 1 --
    D, Y = loadData("smartphone.txt")

    # x1 = D[:, 0]
    # x2 = D[:, 1]

    # plt.scatter(x1, Y)
    # plt.xlabel('x  Original price')
    # plt.ylabel('y  New price')
    # plt.title('Original price VS New price')
    # plt.show()

    # plt.scatter(x2, Y)
    # plt.xlabel('x Age')
    # plt.ylabel('y New price')
    # plt.title('Age VS New price')
    # plt.show()

    # -- 2 --
    #-- Adding a left unity column --
    vector = np.vectorize(np.int_)
    D1 = vector(addOnesColumn(selectSingle(D, 0)))

    # -- 3 --
    rowD1 = D1[0, :]
    hypothesis = [1, 1]
    value = predictValue(rowD1, hypothesis)
    print(value)

if __name__ == '__main__':
    main()

