# No additional 3rd party external libraries are allowed
import numpy as np
from numpy.linalg import norm 

def l2(y, y_hat):
    return norm(y - y_hat)

def l2_grad(y, y_hat):
    z = l2(y, y_hat)
    res = (1/z)*(y - y_hat)
    return res

def cross_entropy(y, y_hat):
    z = np.sum((-y*np.log(y_hat)),((1-y)*-np.log(1-y_hat)))
    return z
    
def cross_entropy_grad(y, y_hat):
    z = np.sum(((1-y)/(1-y_hat)),(-y/y_hat))
    return z
    