import random
import numpy as np
import activationFunctions

class layer:
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = []
    bias = []
    learningRate = 0.1
    inputSize = 0
    
    def __init__(self, inputShape, outputShape, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.initWeights()
        self.initBias()
        self.actFunc = activation
        self.actFuncDerivative = activationD
        
    def reinit(self) -> None:
        self.initWeights()
        self.initBias()
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        output = np.dot(self.weights.T, input)
        output = np.add(output, self.bias)
        output = self.actFunc(output)
        return output
    
    def backPropagation(self, input, output, error):
        gradient =  (error * self.actFuncDerivative(output))
        rValue = self.weights@error
        delta = input@gradient.T
        self.weights = np.subtract(self.weights, self.learningRate * delta)
        self.bias =  np.subtract(self.bias, self.learningRate * gradient)
        return rValue
    
    def initWeights(self):
        temp = []
        for i in range(self.inputShape[0]):
            temp2 = []
            for j in range(self.outputShape[0]):
                temp2.append(random.uniform(-1,1))
            temp.append(temp2)
        self.weights = np.array(temp, ndmin=2)
            
    def initBias(self):
        temp = []
        for j in range(self.outputShape[0]):
            temp.append(random.uniform(-1,1))
        self.bias = np.array(temp, ndmin=2).T
    
    
class softmaxLayer(layer):
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = []
    bias = []
    learningRate = 0.1
    size = 0
    inputSize = 0
    
    actFunc = activation
    actFuncDerivative = activationDerivative
    
    def __init__(self, inputSize) -> None:
        self.size = inputSize
        self.inputSize = inputSize
        
    def reinit(self) -> None:
        pass
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        output = activationFunctions.softmax(input)
        return output
    
    def backPropagation(self, input, output, error):
        pass
    
    def findError(self, prevError):
        return prevError


class flattenLayer(layer):
    def __init__(self, inputShape, outputShape, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.size = outputShape[0]
        
    def reinit(self) -> None:
        pass
    
    def backPropagation(self, input, output, error):
        r = np.reshape(error, self.inputShape)
        return r
        
    def forward(self, input):
        output = np.reshape(input, self.outputShape)
        return output
    
class widenLayer(layer):
    def __init__(self, inputShape, outputShape, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        
    def reinit(self) -> None:
        pass
    
    def findError(self, prevError, output=None):
        #r = np.reshape(prevError, self.inputShape)
        pass
    
    def backPropagation(self, input, output, error):
        r = np.reshape(error, self.inputShape)
        return r
        
    def forward(self, input):
        output = np.reshape(input, self.outputShape)
        return output
    
class maxPoolLayer(layer):
    method = "max"
    def __init__(self, size, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.size = size
        
    def reinit(self) -> None:
        pass
    
    #takes input as matrix, for use inside the network
    def forward(self, input): #recheck later
        m, n = input.shape[:2]
        ky,kx= (2,2)
        pad = True
        _ceil=lambda x,y: int(np.ceil(x/float(y)))

        if pad:
            ny=_ceil(m,ky)
            nx=_ceil(n,kx)
            size=(ny*ky, nx*kx)+input.shape[2:]
            mat_pad=np.full(size,np.nan)
            mat_pad[:m,:n,...]=input
        else:
            ny=m//ky
            nx=n//kx
            mat_pad=input[:ny*ky, :nx*kx, ...]

        new_shape=(ny,ky,nx,kx)+input.shape[2:]

        if self.method=='max':
            result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
        else:
            result=np.nanmean(input.reshape(new_shape),axis=(1,3))

        return result
    
    def backPropagation(self, input, output, error):
        pass
    
    def findError(self, prevError):
        prevError = np.kron(prevError, np.ones((2,2)))
        return prevError
    