import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    h = sigmoid(np.dot(X,theta))

    mt = 1e-7
    cost = (1 / m) * np.sum(-y * np.log(h + mt) - (1 - y) * np.log(-h + 1 + mt), 0)

    grad = (1 / m) * np.dot(X.T, (h - y))

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    # ===========================================================

    return cost, grad
