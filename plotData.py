import matplotlib.pyplot as plt

def plotData(X, y):
    # Create new figure
    plt.figure()
    
    # Find indices of positive and negative examples
    pos = (y == 1)
    neg = (y == 0)
    
    # Plot examples
    plt.plot(X[pos, 0], X[pos, 1], 'r+', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'go', markerfacecolor='r', linewidth=1, markeredgecolor='b', markerfacecolor='g', markersize=7)
    
    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot of Data')
    
    # Show the plot
    plt.show()