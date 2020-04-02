import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    h = sigmoid(np.dot(X,theta))

    cost = (1/m) * np.sum(-y * np.log(h) - (1 - y) * np.log(-h + 1),0)

    grad = (1/m) * np.reshape((h - y),[1,100]).dot(X)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #



    # ===========================================================

    return cost, grad
