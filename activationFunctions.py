import numpy as np

def sigmoid(x):
    x = np.clip( x, -500, 500 )
    return 1 / (1 + np.exp(-x))
    
def sigmoidD(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))
    
def tanh(x):
    return np.tanh(x)

def tanhD(x):
    return (1.0 - np.multiply(np.tanh(x), np.tanh(x)))

def softmax(x):
    return np.exp(x)/sum(np.exp(x))