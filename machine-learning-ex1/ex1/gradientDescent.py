import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )

        m = np.dot(X, theta)
        k = m - y
        k = np.reshape(k,[-1,1])

        ss = k * X

        jj = np.sum(ss,axis=0)

        gradient = jj / X.shape[0]

        theta = theta - alpha * gradient
        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta


        s1 = np.dot(X,theta) - y
        s1 = np.reshape(s1,(-1,1))
        s2 = s1 * X
        s3 = np.sum(s2,axis=0)
        s4 = s3 / m

        theta = theta - alpha * s4


        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
    