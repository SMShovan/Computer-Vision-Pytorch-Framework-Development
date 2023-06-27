# No additional 3rd party external libraries are allowed
import numpy as np

def SGD(model,  train_X, train_y, lr=0.1):
    

    W = model.W  # model

    def get_random_Wj(w):
        # Flatten W to choose the random element from W
        W_flatten = np.squeeze(w.flatten().reshape(1, -1))

        # randomly select the weight and zero out others
        random_choice_j = np.random.choice(W_flatten, size=1, replace=False)
        random_choice_j_value = random_choice_j[0]
        mask = np.ma.masked_equal(w, random_choice_j_value).mask.astype("float")
        W_j = np.multiply(w, mask)

        return W_j

    

    
    for _ in range(25):
        # get random W_j
        w_j1 = get_random_Wj(W[0])
        w_j2 = get_random_Wj(W[1])

        W_j = [w_j1] + [w_j2]

        # compute the gradient of empirical loss
        grad_W_j = model.emp_loss_grad(train_X, train_y, W_j)  # returns 784,

        W[0] = W[0] - lr * grad_W_j[0]
        W[1] = W[1] - lr * grad_W_j[1]

    print("Hello")
    return W
    raise NotImplementedError("SGD not implemented")
    
      