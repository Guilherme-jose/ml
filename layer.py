import random
import numpy as np
import activationFunctions
from neuralNetwork import activation

class layer:
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = []
    bias = []
    learningRate = 0.2
    size = 0
    
    #actFunc = activation
    #actFuncDerivative = activationDerivative
    
    def __init__(self, size, inputSize) -> None:
        self.initWeights(size, inputSize)
        self.initBias(size)
        activation = activationFunctions.tanh
        activationDerivative = activationFunctions.tanhD
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        output = np.matmul(np.transpose(self.weights), input)
        output = np.add(output, self.bias)
        output = self.activation(output)
        return output
    
    def backPropagation(self, input, output, error):
        gradient = self.activationDerivative(output)
        
        self.weights= np.subtract(self.weights, np.matmul(input, np.transpose(self.learningRate * np.multiply(error, gradient))))
        self.bias = np.subtract(self.bias, np.multiply(error, gradient) * self.learningRate)
        
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
            size.append(random.uniform(-1,1))
        self.bias = np.matrix(temp)
    