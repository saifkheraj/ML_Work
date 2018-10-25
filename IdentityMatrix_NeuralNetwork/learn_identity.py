## Run this using ==> python learn_identity.py

import time
import numpy as np

input_size = 8
h1 = 3
output_size = 8

def affine_forward(x, w, b):
    """

    Inputs:
    - x: (N, D)
    - w: (D, M)
    - b: (M,)

    Returns :
    dot product + cache to be used during back prop
    """
    cache=(x,w,b)
    return np.dot(x,w)+b, cache

def affine_backward(gradient_output, cache):
    """
    Inputs:
    - gradient_output
    - cache

    Returns
    backward derivative flow by using gradient output and local derivative
    """

    x, w, b = cache
    dx = np.dot(gradient_output,w.T)

    dw = np.dot(x.T,gradient_output)
    db = np.sum(gradient_output,axis=0)

    return dx,dw,db

def sigmoid(x):
    """
    Inputs:
    - x

    Returns
    transforms x by applying non linearity sigmoid to it.
    """
    cache=x
    x=1/(1+np.exp(-x))
    return x,cache

def sigmoid_backward(gradient_output, cache):
    """

    """
    x=cache
    return (sigmoid(x)[0] * (1-sigmoid(x)[0]))*gradient_output

def cost(x, y,output):
    #sum of squared error without regularization
    Cost=  ( 0.5 * np.sum(np.square(y-output)) ) / float(x.shape[0])
    #print(Cost)
    derivative_cost= (-1.0/x.shape[0]) * (y-output)
    #print(Cost, derivative_cost)
    return Cost,derivative_cost


## Label
y=np.identity(8)
##Input
X=np.identity(8)

##weights for layer 1
w1=np.random.randn(input_size,h1)
b1=np.zeros(h1)

## Final Layer
w2=np.random.randn(h1,output_size)
b2=np.zeros(output_size)

#learning rate, we can increase it if convergence is slow
lr=0.3
history={}

for i in np.arange(0,80000,1):
    ##this could be refactored by putting into single function named as forward
    affine1, cache1_linear=affine_forward(X,w1,b1)
    sig1,cache1_sigmoid=sigmoid(affine1)
    affine2,cache2_linear=affine_forward(sig1,w2,b2)
    sig2,cache2_sigmoid=sigmoid(affine2)

    Cost, der_cost=cost(X,y,sig2)
    if(Cost<0.001):
        break
    history[i]=Cost
    der_sig2=sigmoid_backward(der_cost,cache2_sigmoid)
    dx2,dw2,db2=affine_backward(der_sig2,cache2_linear)
    w2=w2-(lr*dw2)
    b2=b2-(lr*db2)
    der_sig1=sigmoid_backward(dx2,cache1_sigmoid)
    dx1,dw1,db1=affine_backward(der_sig1,cache1_linear)
    w1=w1-(lr*dw1)
    b1=b1-(lr*db1)


print("Final Matrix: ")
print(sig2.round())

### weights are now learnt, To make predictions, given X, we can just do forward pass
