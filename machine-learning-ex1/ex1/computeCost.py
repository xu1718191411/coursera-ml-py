import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    output = np.dot(X, theta)

    cost = (np.sum((output - y) **2)) / 2 * m

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.

    # ==========================================================

    return cost
