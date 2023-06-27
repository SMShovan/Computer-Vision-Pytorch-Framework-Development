# No additional 3rd party external libraries are allowed
import numpy as np

def relu(x):
    return np.maximum(0,x)
    raise NotImplementedError("ReLU function not implemented")

def relu_grad(z):
    # print(type(z))
    # row = len(z)
    # col = len(z[0])

    # for i in range(row):
    #     for j in range(col):
    #         if z[i][j] > 0:
    #             z[i][j] = 1
    #         else:
    #             z[i][j] = 0
    # return z
    return np.greater(z, np.zeros(z.shape, dtype="float64")).astype("float")
 
    raise NotImplementedError("Gradient of ReLU function not implemented")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    raise NotImplementedError("Sigmoid function not implemented")
    
def sigmoid_grad(z):
    # TODO
    return z * (1 - z)
    raise NotImplementedError("Gradient of Sigmoid function not implemented")

def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x), axis=1)).reshape(-1,1)
    raise NotImplementedError("Softmax function not implemented")
    
def softmax_grad(z):
    # TODO
    ilen = len(z[0])
    jlen = len(z[0])
    grad = np.zeros((ilen,jlen))
    for i in range(ilen):
        for j in range(jlen):
            if i == j:
                grad[i,j] = z[0][i] * (1 - z[0][i])
            else:
                grad[i,j] = -1 * z[0][i] * z[0][j]
    return grad

    raise NotImplementedError("Gradient of Softmax function not implemented")

def tanh(x):
    return (np.exp(x) - np.exp(-x))/ (np.exp(x) + np.exp(-x))
    raise NotImplementedError("Tanh function not implemented")

def tanh_grad(z):
    # TODO
    return 1 - tanh(z)**2
    raise NotImplementedError("Gradient of Tanh function not implemented")
