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
    
    def __init__(self, input):
        self.initWeights(input)
        self.initBias(input)
        self.shape = input
    
    def reinit(self, input):
        self.initWeights(input)
        self.initBias(input)
        self.shape = input
        
    #def __init__(self):
    #    self.shape = []

    def addDenseConvolutionLayer(self, size):
        l = layer(size, self.layerList[len(self.layerList - 1)])
        
    def newTrain():
        pass
        
    #receives and returns in list form [-,-,-]
    def guess(self, input):
        inputMatrix = np.matrix(input)
        inputMatrix = np.transpose(inputMatrix)
        outputMatrix = np.matrix([])
        
        it = 0
        for k in self.weights:
            
            outputMatrix = np.matmul(np.transpose(k), inputMatrix)
            outputMatrix = np.add(outputMatrix, self.bias[it])
            outputMatrix = activation(outputMatrix)

            inputMatrix = outputMatrix
            it += 1
        return outputMatrix.tolist()

    def train(self, inputSet, outputSet):
        j = 0
        for i in inputSet:
            outputList = []
            errorList = []

            ################################################
            outputTarget = np.matrix(outputSet[j])
            outputTarget = np.transpose(outputTarget)
            
            inputMatrix = np.matrix(i)
            inputMatrix = np.transpose(inputMatrix)
            
            
            outputList.append(inputMatrix)
            tempMatrix = inputMatrix
            it = 0
            for k in self.weights:
                outputMatrix = np.matmul(np.transpose(k), tempMatrix)
                outputMatrix = np.add(outputMatrix, self.bias[it])
                outputMatrix = activation(outputMatrix)
                
                outputList.append(outputMatrix)
                
                
                tempMatrix = outputMatrix
                it += 1
            
            ##### linear algebra
            
            #if(random.randrange(1,1000) == 999):
                #print(outputTarget, outputMatrix)
            
            
            errorList.append(np.subtract(outputMatrix, outputTarget))
            
            it = 0
            for w in self.weights:
                errorList.append(np.matmul(self.weights[len(self.weights) - 1 - it], errorList[it]))
                it += 1
            errorList.pop()
            it = 0
            for w in self.weights:
                outputGradient = activationD(outputList[it + 1])
                self.weights[it] = np.subtract(self.weights[it], np.matmul(outputList[it], np.transpose(self.learningRate * np.multiply(errorList[len(errorList) - 1 -it], outputGradient)))) ##wtf
                
                self.bias[it] = np.subtract(self.bias[it], np.multiply(errorList[len(errorList) - 1 - it], outputGradient) * self.learningRate)

                it += 1
            j += 1
            
    def initWeights(self, input):
        self.weights.clear()
        for k in range(len(input) - 1):
            temp = []
            for i in range(input[k]):
                temp2 = []
                for j in range(input[k+1]):
                    temp2.append(random.uniform(-1,1))
                temp.append(temp2)
            self.weights.append(np.matrix(temp))

    def initBias(self, input):
        self.bias.clear()
        for k in range(len(input) - 1):
            temp = []
            for i in range(input[k+1]):
                temp.append(random.uniform(-1,1))
            self.bias.append(np.transpose(np.matrix(temp)))

    