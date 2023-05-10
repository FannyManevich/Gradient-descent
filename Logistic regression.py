from main import loadData
from main import addOnesColumn

# Fany Manevich 206116725
# Ilona Grand 316179548

# -- 1 --
# def sigmoid(z):

 # return newMatrix

# -- 2 --
# def predictValue(Example, Hypothesis):
#
#
# # -- 2.1 --
# def computeCostAndGradient(D, Y, Hypothesis):
#
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
    D, Y = loadData("ex2data1.txt")
    x1 = D[:, 0]
    x2 = D[:, 1]

    # -- Adding a left unity column --
    vector = np.vectorize(np.int_)
    D = vector(addOnesColumn(selectSingle(D, 1)))





if __name__ == '__main__':
    main()
