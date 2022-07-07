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
    size = 0
    inputSize = 0
    
    def __init__(self, size, inputSize) -> None:
        self.size = size
        self.inputSize = inputSize
        self.initWeights(size, inputSize)
        self.initBias(size)
        #activation = activationFunctions.tanh
        #activationDerivative = activationFunctions.tanhD
        
    def reinit(self) -> None:
        self.initWeights(self.size, self.inputSize)
        self.initBias(self.size)
        #activation = activationFunctions.tanh
        #activationDerivative = activationFunctions.tanhD
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        output = np.matmul(np.transpose(self.weights), input)
        output = np.add(output, self.bias)
        output = activationFunctions.tanh(output)
        return output
    
    def backPropagation(self, input, output, error):
        gradient = activationFunctions.tanhD(output)
        
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
    