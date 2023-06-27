import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer


class NN2():
    def __init__(self):
        self.layers = [
            Layer(None, relu), 
            Layer(None, sigmoid)]
        

    def nn2(self, x):
        
        
        w1 = self.W[0]
        w2 = self.W[1]
        z1 = linear(x,w1) #(m,k) . (k,1) => (m,1)
        z1_tilda = relu(z1) # (m,1)
        z2 = linear(z1_tilda,w2) # (1,m) . (m,1) => (1,1)
        y_hat = sigmoid(z2)  # (1,1)
        return y_hat
    
        
        raise NotImplementedError("NN2 model not implemented")
        

    def grad(self, x, y, W):
        
        z1 = np.dot(W[0], x.T)#(m,n)
        z1_tilda = relu(z1)#(m,n)
        z2 = np.dot(W[1], z1_tilda)#(1,n)
        y_hat = sigmoid(z2)

        

        # First layer
        grad_yhat_wrt_z2 = sigmoid_grad(y_hat)  # (1,100)
        grad_loss_wrt_yhat = l2_grad(y, y_hat)  # (1,100)
        
        grad_loss_wrt_z2 = np.dot(grad_loss_wrt_yhat, grad_yhat_wrt_z2.T)  # (1,100) . (100,1) =>(1,1)
        
        grad_z2_wrt_z1tilda = W[1].T  # (200,1)
        
        grad_loss_wrt_z1tilda = np.multiply(grad_loss_wrt_z2.T, grad_z2_wrt_z1tilda)  # (1,1) . (200,1) => (200,1)
        

        grad_z1tilda_wrt_z1 = relu_grad(z1)  # (200,100)
        
        grad_loss_wrt_z1 = np.multiply(grad_loss_wrt_z1tilda.squeeze(), grad_z1tilda_wrt_z1)  # (200,1) . (200,100) => (200,100)
        
        grad_z1_wrt_w1 = linear_grad(x)  # (100,784)
        grad_loss_wrt_w1 = np.dot(np.array([list(grad_loss_wrt_z1)]).T, np.array([list(grad_z1_wrt_w1)]))  # (200,100) . (100,784) => (200,784)

        # Second Layer
        grad_z2_wrt_w2 = z1_tilda  # (200,100)
        grad_loss_wrt_w2 = np.multiply(grad_loss_wrt_z2, grad_z2_wrt_w2)  # (1,1) . (200,100) => (200,100)

        return [grad_loss_wrt_w1, grad_loss_wrt_w2]
        raise NotImplementedError("NN2 gradient (backpropagation) not implemented") 
        

    def emp_loss_grad(self, train_X, train_y, W):
        
        
        gradient_loss1 = np.zeros_like(W[0])
        gradient_loss2 = np.zeros_like(W[1])
        

        for it in range(len(train_X)):
            
            gradient_loss1 = gradient_loss1+ self.grad(train_X[it], train_y[it], W)[0]
            gradient_loss2 = gradient_loss2+ self.grad(train_X[it], train_y[it], W)[1]

        gradient_loss1 = np.sum(gradient_loss1)/ train_X.shape[0]
        gradient_loss2 = np.sum(gradient_loss2)/train_X.shape[0]

        grad = [gradient_loss1, gradient_loss2]

        return grad
        raise NotImplementedError("NN2 Emperical Loss grad not implemented")

