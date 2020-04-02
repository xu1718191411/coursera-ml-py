import numpy as np


def sigmoid(z):
    g = np.zeros(z.size)

    # ===================== Your Code Here =====================
    # Instructions : Compute the sigmoid of each value of z (z can be a matrix,
    #                vector or scalar
    #
    # Hint : Do not import math

    result = 1 + np.exp(-z)
    result = 1 / result
    return result



sigmoid(np.array([-50]))
