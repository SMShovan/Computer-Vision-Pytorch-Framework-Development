# No additional 3rd party external libraries are allowed
import numpy as np

def dropout(x, p, mode='test'):
    '''
    Output : should return a tuple containing 
     - z : output of the dropout
     - p : Dropout param
     - mode : 'test' or 'train'
     - mask : 
      - in train mode, it is the dropout mask
      - in test mode, mask will be None.
    
    sample output: (z=, p=0.5, mode='test',mask=None)
    '''
    if mode == 'train':
        mask = np.random.binomial(1, p, (x.shape[0], x.shape[1]))/p
        z = x*mask
        return z, mask

    elif mode == 'test':
        z = x.copy()
        return z, np.ones(x.shape)


def dropout_grad(z, mode='train'):
    if mode == 'train':
        mask = z[1]
        return mask
    
    elif mode == 'test':
        return np.ones(z.shape)