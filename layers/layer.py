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
    