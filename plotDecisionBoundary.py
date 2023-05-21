import numpy as np
import matplotlib.pyplot as plt
from fontTools.mtiLib import mapFeature


def plotDecisionBoundary(theta, X, y):
    # Plot Data
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    if theta[0] == 0:
        theta[0] = 0.001
    if theta[1] == 0:
        theta[1] = 0.001
    if theta[2] == 0:
        theta[2] = 0.001

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, color='purple')
        plt.legend(loc='upper right')
        plt.show()

        # Legend, specific for the exercise
        plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
        plt.xlim([20, 100])
        plt.ylim([20, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[i], v[j]).dot(theta)

        plt.contour(u, v, z, levels=[0], colors='purple', linewidths=2)

    plt.show()
