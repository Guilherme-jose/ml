import math
import random
import numpy as np
import activationFunctions
import layer

def activation(x):
    return activationFunctions.tanh(x)

def activationD(x):
    return activationFunctions.tanhD(x)
    
class NeuralNetwork:
    learningRate = 0.2
    
    weights = []
    bias = []
    shape = []
    activations = []
    activationsD = []
    layerList = []
    inputSize = 0
    
    batchSize = 1.0 #how much of the dataset to include in the training set for a given iteration
    
    def __init__(self, input):
        self.inputSize = input
        
    def reinit(self):
        for i in self.layerList:
            i.reinit()
        
    def addDenseLayer(self, size, activationFunction=activationFunctions.tanh):
        prevSize = self.inputSize
        if(len(self.layerList) > 0):
            prevSize = self.layerList[len(self.layerList) - 1].size
        l = layer.layer(size, prevSize)
        self.layerList.append(l)
    
    def guess(self, input):
        inputMatrix = np.matrix(input)
        inputMatrix = np.transpose(inputMatrix)
        
        outputMatrix = inputMatrix
        for it in range(len(self.layerList)):
            outputMatrix = self.layerList[it].forward(outputMatrix)
            
        return outputMatrix.tolist()
        
    def train(self, inputSet, outputSet, epochs):
        for epoch in range(epochs):
            for k in range(len(inputSet)):#range(math.floor(len(inputSet)*self.batchSize)):
                j = random.randrange(0, len(outputSet))
                
                outputList = []
                errorList = []

                outputTarget = np.transpose(np.matrix(outputSet[j]))
                inputMatrix = np.transpose(np.matrix(inputSet[j]))
                
                outputList.append(inputMatrix)
                outputMatrix = inputMatrix
                
                for it in range(len(self.layerList)):
                    outputMatrix = self.layerList[it].forward(outputMatrix)
                    outputList.append(outputMatrix)
                
                errorList.append(np.subtract(outputMatrix, outputTarget))
                
                for i in range(len(self.layerList)):
                    errorList.append(self.layerList[len(self.layerList) - i - 1].findError(errorList[i]))
                errorList.pop()
                
                for i in range(len(self.layerList)):
                    self.layerList[i].backPropagation(outputList[i], outputList[i + 1], errorList[len(errorList) - i - 1])
            if(epoch%1000==0):
                print("epoch:", epoch)
                
    