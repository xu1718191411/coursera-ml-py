import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    # ===========================================================

    h = sigmoid(np.dot(X, theta))
    sm = 1e-7
    cost = (1 / m) * np.sum(y * -1 * np.log(h+ sm) + (1 - y) * -1 * np.log(-h + 1 + sm), 0) + (lmd / (2 * m)) * np.sum(theta ** 2)
    grads = (1 / m) * np.dot(X.T, (h - y))
    grad[0] =  grads[0]
    grad[1:] = grads[1:] + (lmd / m) * theta[1:]



    return cost, grad
