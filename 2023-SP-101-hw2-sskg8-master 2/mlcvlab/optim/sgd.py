# No additional 3rd party external libraries are allowed
import numpy as np


def SGD(model, train_x_batches, train_y_batches, lr, R):
    for _ in range(R):
        for mini_batch_x, mini_batch_y in zip(train_x_batches, train_y_batches):
            # Forward
            A, cache = model.nn4(mini_batch_x) #Returns y4 and dictionary of all the elements

            # Backward
            grads = model.emp_loss_grad(mini_batch_x, mini_batch_y, A, cache)

            # Update

            model.layers[3].W = model.layers[3].W - lr * grads[3]
            model.layers[2].W = model.layers[2].W - lr * grads[2]
            model.layers[1].W = model.layers[1].W - lr * grads[1]
            model.layers[0].W = model.layers[0].W - lr * grads[0]

    return model


