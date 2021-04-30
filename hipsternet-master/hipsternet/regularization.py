import numpy as np
import hipsternet.constant as c
from numpy import linalg as LA


def l2_reg(W, lam=1e-3):
    return .5 * lam * np.sum(W * W)


def dl2_reg(W, lam=1e-3):
    return lam * W


def l1_reg(W, lam=1e-3):
    return lam * np.sum(np.abs(W))


def dl1_reg(W, lam=1e-3):
    return lam * W / (np.abs(W) + c.eps)
    

# parameters as a weight vector representing a layer in nn, max_value as a hyper-param constrainting the maximum L2 norm of the weight vector
def limit_norm(parameters, max_val=3, eps=1e-8):
    norm = LA.norm(parameters)
    desired = max(min(norm, max_val), 0)
    parameters *= desired / (eps + norm)
    return parameters