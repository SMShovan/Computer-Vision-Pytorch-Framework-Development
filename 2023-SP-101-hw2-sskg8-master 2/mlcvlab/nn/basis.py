# No additional 3rd party external libraries are allowed
import numpy as np

def linear(x, W):
    Y = np.dot(W,x)
    return Y
    #raise NotImplementedError("Linear function not implemented")
    
def linear_grad(x):
    return x
    #raise NotImplementedError("Gradient of Linear function not implemented")

def radial(x, W):
    # TODO
    raise NotImplementedError("Radial Basis function not implemented")
    
def radial_grad(loss_grad_y, x, W):
    # TODO
    raise NotImplementedError("Gradient of Radial Basis function not implemented")