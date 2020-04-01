import numpy as np


def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #

    XT = X.T
    s1 = np.dot(XT, X)
    s2 = np.linalg.pinv(s1)
    s2 = np.dot(s2, XT)
    theta = np.dot(s2, y.reshape(47,1))
    return theta
