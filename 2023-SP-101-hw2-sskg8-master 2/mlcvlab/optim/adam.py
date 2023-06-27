# No additional 3rd party external libraries are allowed
import numpy as np

def Adam(model,train_X,train_y):
    delta = 1e-8
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    final_w = model.W
    for i in range(2):
        s = np.random.random(1).item()
        m = np.random.random(1).item()
        for _ in range(10):
            W_m = [W+ beta1 * m for W in final_w]
            m = beta1*m+(1-beta1)*model.emp_loss_grad(train_X,train_y,W_m)[i]
            s = beta2*s+(1-beta2)*np.dot(model.emp_loss_grad(train_X, train_y,final_w)[i].T,model.emp_loss_grad(train_X,       train_y,final_w)[i])
            final_w[i]=final_w[i]-((m*alpha)/(np.sqrt(s)+delta))
      
    return final_w


    