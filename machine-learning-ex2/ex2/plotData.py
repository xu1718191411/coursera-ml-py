import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #

    posIndex = np.where(y == 1)
    negIndex = np.where(y == 0)

    posX = X[posIndex]
    negX = X[negIndex]

    plt.scatter(x=posX[:,0],y=posX[:,1],marker="o",c="b")
    plt.scatter(x=negX[:,0],y=negX[:,1],marker="x",c="r")
