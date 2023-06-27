# No additional 3rd party external libraries are allowed
import numpy as np
from numba import cuda

@cuda.jit

def sync_sgd(model, X_train_batches, y_train_batches, lr=0.1, R=100):
    '''
    Compute gradient estimate of emp loss on each mini batch in-parallel using GPU blocks/threads.
    Wait for all results and aggregate results by calling cuda.synchronize(). For more details, refer to https://thedatafrog.com/en/articles/cuda-kernel-python
    Compute update step synchronously
    '''
    i1, i2 = cuda.grid(2)
    for _ in range(R):
        for mini_batch_x, mini_batch_y in zip(train_x_batches, train_y_batches):
            # Forward
            A, cache = model.nn4(mini_batch_x[i1][i2])

            # Backward
            grads = model.emp_loss_grad(mini_batch_x[i1][i2], mini_batch_y[i1][i2], A, cache)

            # wait for all to finish
            cuda.synchronize()

            # Update
            model.layers[3].W = model.layers[3].W - lr * grads[3]
            model.layers[2].W = model.layers[2].W - lr * grads[2]
            model.layers[1].W = model.layers[1].W - lr * grads[1]
            model.layers[0].W = model.layers[0].W - lr * grads[0]

    return model[i1][i2]