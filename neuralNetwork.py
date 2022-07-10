import math
import random
import numpy as np
import activationFunctions
import layer
import pygame

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
    
    def __init__(self, input, inputY=1):
        self.inputSize = input
        self.inputSizeT = inputY
        
    def reinit(self):
        for i in self.layerList:
            i.reinit()
        
    def addDenseLayer(self, size, activationFunction=activationFunctions.tanh, activationFunctionD=activationFunctions.tanhD):
        prevSize = self.inputSize
        if(len(self.layerList) > 0):
            prevSize = self.layerList[len(self.layerList) - 1].size
        l = layer.layer(size, prevSize, activationFunction, activationFunctionD)
        self.layerList.append(l)
        
    def addSoftmaxLayer(self):
        prevSize = self.inputSize
        if(len(self.layerList) > 0):
            prevSize = self.layerList[len(self.layerList) - 1].size
        l = layer.softmaxLayer(prevSize)
        self.layerList.append(l)
        
    def addConvLayer(self, size, kernelSize):
        l = layer.kernelLayer(kernelSize, size)
        self.layerList.append(l)
        
    def addFlattenLayer(self, size, outSize):
        l = layer.flattenLayer(outSize, size)
        self.layerList.append(l)
    
    def addWidenLayer(self, size):
        l = layer.widenLayer(size)
        self.layerList.append(l)
        
    def addMaxPoolLayer(self, size):
        l = layer.maxPoolLayer(size)
        self.layerList.append(l)
        
    def guess(self, input):
        inputMatrix = np.matrix(input)
        inputMatrix = np.transpose(inputMatrix)
        
        outputMatrix = inputMatrix
        for it in range(len(self.layerList)):
            outputMatrix = self.layerList[it].forward(outputMatrix)
            
        return outputMatrix.tolist()
        
    def train(self, inputSet, outputSet, epochs, mode=""):
        iterations = 0
        for epoch in range(epochs):
            for k in range(len(inputSet)):#range(math.floor(len(inputSet)*self.batchSize)):
                j = random.randrange(0, len(outputSet)-1)
                
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
                    
                    
                iterations += 1
                if(iterations%1000==0): 
                    print("iterations:", iterations)
                    if(mode == "classifier"):
                        pass
                        self.testClassifier(inputSet, outputSet)
            print("epoch:", epoch)
            if(mode == "classifier"):
                        self.testClassifier(inputSet, outputSet)
    
    def testClassifier(self, trainingSet, trainingOutput):
        t = max(1, math.floor(len(trainingSet)/100))
        count = 0
        for i in range(t): 
            j = random.randrange(len(trainingSet) - 1)
            if(np.argmax(self.guess(trainingSet[j])) == np.argmax(trainingOutput[j])):
                count += 1
        print(count*100/t, "% " + "over " + str(t) + (" tests"))
    