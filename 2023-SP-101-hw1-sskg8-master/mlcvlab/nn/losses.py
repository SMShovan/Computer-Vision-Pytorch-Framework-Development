# No additional 3rd party external libraries are allowed
import numpy as np
from numpy.linalg import norm 

def l2(y, y_hat):
    # TODO
    return np.sqrt(np.sum(np.power((y-y_hat),2)))
    raise NotImplementedError("l2 loss function not implemented")

def l2_grad(y, y_hat):
    # TODO
    return (y-y_hat)/l2(y, y_hat)
    raise NotImplementedError("Gradiant of l2 loss function not implemented")

def cross_entropy(A, Y):
    # TODO
    return np.mean(-1 * A * np.log(Y) - (1 - A) * np.log(1 - Y))
    raise NotImplementedError("Cross entropy loss function not implemented")
    
def cross_entropy_grad(y, y_hat):
    # TODO
    return (1 - y) / (1 - y_hat) - (y / y_hat)
    raise NotImplementedError("Gradiant of Cross entropy loss function not implemented")
    