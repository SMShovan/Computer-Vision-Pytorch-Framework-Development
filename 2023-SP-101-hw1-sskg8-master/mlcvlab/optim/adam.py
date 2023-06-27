# No additional 3rd party external libraries are allowed
import numpy as np


def Adam(model, train_X, train_y):
   
    W = model.W  # list of 784 X 1 numpy array
    

    # Hyperparameters
    delta = 10 ** -8
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999

    
    for i in range(2):
        # initialize random momentum(m) and rmsprop(s)
        m = np.random.random(1).item()
        s = np.random.random(1).item()

        for r in range(100):
            Wm = [w + beta1 * m for w in W]
            m = beta1 * m + (1 - beta1) * model.emp_loss_grad(train_X, train_y, Wm)[i]
            s = beta2 * s + (1 - beta2) * np.dot(model.emp_loss_grad(train_X, train_y, W)[i].T,
                                                    model.emp_loss_grad(train_X, train_y, W)[i])
            W[i] = W[i] - m * alpha / (np.sqrt(s) + delta)
        return W
    

