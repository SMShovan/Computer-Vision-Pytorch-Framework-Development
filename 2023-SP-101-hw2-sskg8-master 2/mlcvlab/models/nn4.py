import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from mlcvlab.nn.batchnorm import batchnorm, batchnorm_grad
from mlcvlab.nn.dropout import dropout, dropout_grad
from .base import Layer


class NN4():
    def __init__(self, use_batchnorm=False, dropout_param=0):
        self.layers = [
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, sigmoid)]

        self.use_batchnorm = use_batchnorm

        # used in dropout implementation
        self.dropout_param = dropout_param

    def nn4(self, x, mode='train'):
        W1 = self.layers[0].W
        W2 = self.layers[1].W
        W3 = self.layers[2].W
        W4 = self.layers[3].W

        gamma1 = 0.3
        gamma2 = 0.2
        gamma3 = 0.1
        beta1 = 0.1
        beta2 = 0.2
        beta3 = 0.4
        eps = 0.01

        cache = {
            "gamma1": gamma1,
            "gamma2": gamma2,
            "gamma3": gamma3,
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
            "eps": eps,
            "W1": W1,
            "W2": W2,
            "W3": W3,
            "W4": W4
        }
        if self.use_batchnorm:
            z1 = linear(x.T, W1)#(300,784)*(784,100)=>300,100
            y1 = self.layers[0].activation(z1)#300,100
            b1 = batchnorm(y1, gamma1, beta1, eps)[0]#300,100
            x1, mask1 = dropout(b1, self.dropout_param, mode)#300,100
            cache["z1"] = z1
            cache["y1"] = y1
            cache["b1"] = b1
            cache["x1"] = x1
            cache["mask1"] = mask1

            z2 = linear(x1, W2)#(150,300)*(300,100)=>150,100
            y2 = self.layers[1].activation(z2)#150,100
            b2 = batchnorm(y2, gamma2, beta2, eps)[0]#150,100
            x2, mask2 = dropout(b2, self.dropout_param, mode)#150,100
            cache["z2"] = z2
            cache["y2"] = y2
            cache["b2"] = b2
            cache["x2"] = x2
            cache["mask2"] = mask2

            z3 = linear(x2, W3)#(50,150)*(150,100)=>50,100
            y3 = self.layers[2].activation(z3)#50,100
            b3 = batchnorm(y3, gamma3, beta3, eps)[0]#50,100
            x3, mask3 = dropout(b3, self.dropout_param, mode)#50,100
            cache["z3"] = z3
            cache["y3"] = y3
            cache["b3"] = b3
            cache["x3"] = x3
            cache["mask3"] = mask3

            z4 = linear(x3, W4)#(1,50)*(50,100)=>1,100
            y4 = self.layers[3].activation(z4)#1,100
            x4, mask4 = dropout(y4, self.dropout_param, mode)#1,100
            cache["z4"] = z4
            cache["y4"] = y4
            cache["x4"] = x4
            cache["mask4"] = mask4

        else:
            z1 = linear(x.T, W1)#300,100
            y1 = self.layers[0].activation(z1)
            x1, mask1 = dropout(y1, self.dropout_param, mode)
            cache["z1"] = z1
            cache["y1"] = y1
            cache["x1"] = x1
            cache["mask1"] = mask1

            z2 = linear(x1, W2)#150,100
            y2 = self.layers[1].activation(z2)
            x2, mask2 = dropout(y2, self.dropout_param, mode)
            cache["z2"] = z2
            cache["y2"] = y2
            cache["x2"] = x2
            cache["mask2"] = mask2

            z3 = linear(x2, W3)#50,100
            y3 = self.layers[2].activation(z3)
            x3, mask3 = dropout(y3, self.dropout_param, mode)
            cache["z3"] = z3
            cache["y3"] = y3
            cache["x3"] = x3
            cache["mask3"] = mask3

            z4 = linear(x3, W4)#1,100
            y4 = self.layers[3].activation(z4)
            x4, mask4 = dropout(y4, self.dropout_param, mode)#EXTRA
            cache["z4"] = z4
            cache["y4"] = y4
            cache["x4"] = x4
            cache["mask4"] = mask4

        return y4, cache

    def grad(self, x, y, y_hat, cache):
        gamma1 = cache["gamma1"]
        gamma2 = cache["gamma2"]
        gamma3 = cache["gamma3"]
        beta1 = cache["beta1"]
        beta2 = cache["beta2"]
        beta3 = cache["beta3"]
        eps = cache["eps"]
        W1 = cache["W1"]
        W2 = cache["W2"]
        W3 = cache["W3"]
        W4 = cache["W4"]
        z1 = cache["z1"]
        y1 = cache["y1"]
        b1 = cache["b1"]
        x1 = cache["x1"]
        mask1 = cache["mask1"]
        z2 = cache["z2"]
        y2 = cache["y2"]
        b2 = cache["b2"]
        x2 = cache["x2"]
        mask2 = cache["mask2"]
        z3 = cache["z3"]
        y3 = cache["y3"]
        b3 = cache["b3"]
        x3 = cache["x3"]
        mask3 = cache["mask3"]
        z4 = cache["z4"]
        y4 = cache["y4"]
        x4 = cache["x4"]
        mask4 = cache["mask4"]

        if self.use_batchnorm:
            
            grad_loss_x4 = l2_grad(y, y_hat) # 100,1* 1,100=>100,100 
            grad_loss_y4 = np.dot(grad_loss_x4, mask4.T)#100,100*100,1=>100,1
            grad_loss_z4 = np.multiply(grad_loss_y4, sigmoid_grad(y_hat.T)) # 100,1*100,1=>100,1
            grad_loss_w4 = np.dot(grad_loss_z4.T, x3.T) # 1,100 * 100,50 = 1,50

            grad_loss_x3 = np.dot(grad_loss_z4, linear_grad(W4)) #100,1*1,50=>100,50
            grad_loss_b3 = np.multiply(grad_loss_x3.T, mask3) #50,100*50,100=>50,100
            grad_loss_y3, grad_loss_gamma3, grad_loss_beta3 = batchnorm_grad(grad_loss_b3, y3, gamma3, beta3, eps) #50,100
            grad_loss_z3 = np.multiply(grad_loss_y3, relu_grad(y3))#50,100*50,100=>50,100
            grad_loss_w3 = np.dot(grad_loss_z3, x2.T)#50,100*100,150=>50,150

            grad_loss_x2 = np.dot(grad_loss_z3.T, linear_grad(W3)) #100,50*50,150=>100,150
            grad_loss_b2 = np.multiply(grad_loss_x2.T, mask2)#150,100
            grad_loss_y2, grad_loss_gamma2, grad_loss_beta2 = batchnorm_grad(grad_loss_b2, y2, gamma2, beta2, eps)#150,100
            grad_loss_z2 = np.multiply(grad_loss_y2, relu_grad(y2)) #150,100*150,100=>150,100
            grad_loss_w2 = np.dot(grad_loss_z2, x1.T) #150,100*100,300=>150,300

            grad_loss_x1 = np.dot(grad_loss_z2.T, linear_grad(W2))#100,150*150,300=>100,300            
            grad_loss_b1 = np.multiply(grad_loss_x1.T, mask1) #300,100
            grad_loss_y1, grad_loss_gamma1, grad_loss_beta1 = batchnorm_grad(grad_loss_b1, y1, gamma1, beta1, eps) #300,100
            grad_loss_z1 = np.multiply(grad_loss_y1, relu_grad(y1)) #300,100*300,100=>300,100
            grad_loss_w1 = np.dot(grad_loss_z1, x) #300,100*100,784=>300,784
            
            return [grad_loss_w4, grad_loss_w3,grad_loss_w2,grad_loss_w1]

        else:
            grad_loss_x4 = l2_grad(y, y_hat)  #100,100
            grad_loss_y4 = np.dot(grad_loss_x4, mask4.T)#100,100*100,1=>100,1
            grad_loss_z4 = np.dot(grad_loss_y4, sigmoid_grad(y_hat.T)) # 100,1*100,1=>100,1
            grad_loss_w4 = np.dot(grad_loss_z4.T, x3.T) # 1,100 * 100,50 = 1,50

            grad_loss_x3 = np.dot(grad_loss_z4, linear_grad(W4)) #100,1*1,50=>100,50
            grad_loss_y3 = np.multiply(grad_loss_x3.T, mask3) #50,100
            grad_loss_z3 = np.multiply(grad_loss_y3, relu_grad(y3))#50,100*50,100=>50,100
            grad_loss_w3 = np.dot(grad_loss_z3, x2.T)#50,100*100,150=>50,150

            grad_loss_x2 = np.dot(grad_loss_z3.T, linear_grad(W3))#100,50*50,150=>100,150
            grad_loss_y2 = np.multiply(grad_loss_x2.T, mask2)  #150,100
            grad_loss_z2 = np.multiply(grad_loss_y2, relu_grad(y2))  #150,100*150,100=>150,100
            grad_loss_w2 = np.dot(grad_loss_z2, x1.T) #150,100*100,300=>150,300

            grad_loss_x1 = np.dot(grad_loss_z2.T, linear_grad(W2)) #100,150*150,300=>100,300
            grad_loss_y1 = np.multiply(grad_loss_x1.T, mask1)  #300,100
            grad_loss_z1 = np.multiply(grad_loss_y1, relu_grad(y1))  #300,100*300,100=>300,100
            grad_loss_w1 = np.dot(grad_loss_z1, x)  #300,100*100,784=>300,784

            return [grad_loss_w4, grad_loss_w3,grad_loss_w2,grad_loss_w1]
        
    def emp_loss_grad(self, train_X, train_y, A, cache):
            grad = [np.sum(grad_loss) / train_X.shape[0] for grad_loss in self.grad(train_X, train_y, A, cache)]
            print(grad)
            return grad

