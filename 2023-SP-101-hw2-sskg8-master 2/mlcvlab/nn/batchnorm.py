# No additional 3rd party external libraries are allowed
import numpy as np

from mlcvlab.nn.losses import l2_grad


def batchnorm(x, gamma, beta, eps):
    '''
    Input:
    - x: Data of shape (M, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - mode: 'train' or 'test'; required
    output:
    - y
    Suggestion: When you return the output, also return the necessary intermediate values (i.e. mu, variance, x_norm) needed in gradient computation
    '''

    mue = np.mean(x)
    var = np.var(x)
    x_norm = (x - mue) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    return out, x_norm, mue, var


def batchnorm_grad(y, x, gamma, beta, eps):
    y, x_normalized, mue, var = batchnorm(x, gamma, beta, eps)
    
    m = x.shape[0]
    
    grad_loss_wrt_xhat = y * gamma
    grad_loss_wrt_var = np.sum(grad_loss_wrt_xhat * (x - mue) * (-0.5) * (var + eps) ** (-3/2), axis=0)
    grad_loss_wrt_mue = np.sum(-grad_loss_wrt_xhat / np.sqrt(var + eps), axis=0)
    grad_loss_wrt_xi = grad_loss_wrt_xhat / np.sqrt(var + eps) + (grad_loss_wrt_var * 2 * (x - mue) / m) + (grad_loss_wrt_mue / m)
    
    grad_loss_wrt_gamma = np.sum(y * x_normalized, axis=0)
    
    grad_loss_wrt_beta = np.sum(y, axis=0)


    return grad_loss_wrt_xi, grad_loss_wrt_gamma, grad_loss_wrt_beta
