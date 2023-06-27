# No additional 3rd party external libraries are allowed
import numpy as np

def relu(x):
    return np.maximum(0.0,x)
   
def relu_grad(z):
    for i in range(0,z.shape[0]):
        for j in range(0,z.shape[1]):
            if z[i][j]>0:
                z[i][j]=1
            else:
                z[i][j]=0
    return z

def sigmoid(x):
    y = 1. / (1. + np.exp(-x))
    return y
    #raise NotImplementedError("Sigmoid function not implemented")
    
def sigmoid_grad(y):
    z = y * (1-y)
    return z

def softmax(x):
    s = np.zeros(x.shape)
    for i in range(x.shape[0]):
        term1 = np.exp(x[i])
        s[i] = term1/np.sum(term1)
    return s
    
def softmax_grad(z):
    a = np.zeros([z.shape[1], z.shape[1]], dtype = float) 

    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            if i == j:
                a[i][j]= z[0][j]*(1-z[0][j]) 
            else:
                a[i][j]= -z[0][j]*z[0][i]
    return a

def tanh(x):
    num= np.exp(x)-np.exp(-x)
    den= np.exp(x)+np.exp(-x)
    tanh= np.divide(num,den)
    return tanh

def tanh_grad(y):
    y = tanh(y)
    z = (1-np.square(y))
    return z
