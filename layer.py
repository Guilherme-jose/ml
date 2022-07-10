import random
import numpy as np
import activationFunctions
from scipy import signal

class layer:
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = []
    bias = []
    learningRate = 0.1
    size = 0
    inputSize = 0
    
    def __init__(self, size, inputSize, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.size = size
        self.inputSize = inputSize
        self.initWeights(size, inputSize)
        self.initBias(size)
        self.actFunc = activation
        self.actFuncDerivative = activationD
        
    def reinit(self) -> None:
        self.initWeights(self.size, self.inputSize)
        self.initBias(self.size)
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        output = np.matmul(np.transpose(self.weights), input)
        output = np.add(output, self.bias)
        output = self.actFunc(output)
        return output
    
    def backPropagation(self, input, output, error):
        gradient = self.actFuncDerivative(output)
        
        self.weights = np.subtract(self.weights, np.matmul(input, np.transpose(self.learningRate * np.multiply(error, gradient))))
        self.bias = np.subtract(self.bias, np.multiply(error, gradient) * self.learningRate * 0.5)
    
    def findError(self, prevError):
        return np.matmul(self.weights, prevError)
    
    def initWeights(self, size, inputSize):
        temp = []
        for i in range(inputSize):
            temp2 = []
            for j in range(size):
                temp2.append(random.uniform(-1,1))
            temp.append(temp2)
        self.weights = np.matrix(temp)
            
    def initBias(self, size):
        temp = []
        for j in range(size):
            temp.append(random.uniform(-1,1))
        self.bias = np.transpose(np.matrix(temp))
    
    
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

class kernelLayer(layer):
    kernelSize = []
    kernelNumber = 1
    def __init__(self, kernelSize, size, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.size = size
        self.kernelSize = kernelSize
        self.initWeights(kernelSize, kernelSize)
        self.actFunc = activation
        self.actFuncDerivative = activationD
        
    def reinit(self) -> None:
        self.initWeights(self.kernelSize, self.kernelSize)
    
    def backPropagation(self, input, output, error):
        gradient = self.actFuncDerivative(output)
        error =  np.multiply(error, gradient)
        if((input.size%self.kernelSize) != 0):
            input = np.reshape(-1,1)
            error = np.reshape(-1,1)
            shapedInput = np.pad(input.astype(float), (0, (self.kernelSize)*((input.size//self.kernelSize)+1) - input.size), mode='constant', constant_values=0).reshape(self.kernelSize,-1)
            shapedError = np.pad(error.astype(float), (0, (self.kernelSize)*((error.size//self.kernelSize)+1) - error.size), mode='constant', constant_values=0).reshape(self.kernelSize,-1)
        else:
            shapedInput = np.reshape(input, (self.kernelSize,-1))
            shapedError = np.reshape(error, (self.kernelSize,-1))
        
        
        self.weights = np.subtract(self.weights, np.matmul(shapedInput, np.transpose(self.learningRate * shapedError)))
        
    def forward(self, input):
        output = signal.convolve2d(input, self.weights, mode="same")
        output = self.actFunc(output)
        return input
    
    def findError(self, prevError):
        return prevError

class flattenLayer(layer):
    def __init__(self, outSize, size, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.size = outSize
        self.inSize = size
        self.inputSize = 0
        
    def reinit(self) -> None:
        pass
    
    def findError(self, prevError):
        r = np.reshape(prevError, (self.inSize, -1))
        return r
    
    def backPropagation(self, input, output, error):
        pass
        
    def forward(self, input):
        output = np.reshape(input, (-1,1))
        return output
    
class widenLayer(layer):
    def __init__(self, size, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.size = size
        self.inputSize = 0
        
    def reinit(self) -> None:
        pass
    
    def findError(self, prevError):
        r = np.reshape(prevError, (1,-1))
        return r
    
    def backPropagation(self, input, output, error):
        pass
        
    def forward(self, input):
        output = np.reshape(input, (self.size, -1))
        return output
    
class maxPoolLayer(layer):
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

        #if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
        #else:
           # result=np.nanmean(input.reshape(new_shape),axis=(1,3))

        return result
    
    def backPropagation(self, input, output, error):
        pass
    
    def findError(self, prevError):
        prevError = np.kron(prevError, np.ones((2,2)))
        return prevError
    