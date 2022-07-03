import math
import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoidD(x):
    return np.multiply(sigmoid(x), sigmoid(1 - x))

class NeuralNetwork:
    learningRate = 0.2   
    weights = []
    bias = []
    shape = []

    def __init__(self, input):
        self.initWeights(input)
        self.initBias(input)
        self.shape = input

    #receives and returns in list form [-,-,-]
    def guess(self, input):
        inputMatrix = np.matrix(input)
        inputMatrix = np.transpose(inputMatrix)
    
        it = 0
        for k in self.weights:
            
            outputMatrix = np.matmul(np.transpose(k), inputMatrix)
            outputMatrix = np.add(outputMatrix, self.bias[it])
            outputMatrix = sigmoid(outputMatrix)

            inputMatrix = outputMatrix
            it += 1
        
        return outputMatrix.tolist()

    def train(self, inputSet, outputSet):
        j = 0
        for input in inputSet:
            outputList = []
            errorList = []
            
            outputTarget = np.matrix(outputSet[j])
            outputTarget = np.transpose(outputTarget)
            inputMatrix = np.matrix(input)
            inputMatrix = np.transpose(inputMatrix)
            
            outputList.append(inputMatrix)
            
            it = 0
            for k in self.weights:
                outputMatrix = np.matmul(np.transpose(k), inputMatrix)
                outputMatrix = np.add(outputMatrix, self.bias[it])
                outputMatrix = sigmoid(outputMatrix)
                
                outputList.append(outputMatrix)
                
                
                inputMatrix = outputMatrix
                it += 1
            outputList.reverse()
            
            reverseWeights = self.weights
            reverseWeights.reverse()
            
            reverseBias = self.bias
            reverseBias.reverse()
            
            errorList.append(np.subtract(outputList[0], outputTarget))

            it = 0
            for k in reverseWeights:
                #print (k, errorList[it])
                errorList.append(np.dot((k), errorList[it]))    
                it += 1
            

            ##### linear algebra
            
            k = 0
            for i in reverseWeights:
                gradient = sigmoidD(outputList[k])
                
                delta = np.matmul(outputList[k+1], np.transpose(self.learningRate * np.multiply(errorList[k], gradient)))
                i = np.subtract(i, delta)
                #i = sigmoid(i)
                
                reverseBias[k] = np.subtract(reverseBias[k], np.multiply(errorList[k], gradient) * self.learningRate)
                reverseBias[k] = sigmoid(reverseBias[k])

                k += 1
            j += 1
            
            reverseWeights.reverse()
            self.weights = reverseWeights
            
            reverseBias.reverse()
            self.bias = reverseBias
            

            
    def initWeights(self, input):
        for k in range(len(input) - 1):
            temp = []
            for i in range(input[k]):
                temp2 = []
                for j in range(input[k+1]):
                    temp2.append(random.uniform(-1,1))
                temp.append(temp2)
            self.weights.append(np.matrix(temp))
            

    def initBias(self, input):
        for k in range(len(input) - 1):
            temp = []
            for i in range(input[k+1]):
                temp.append(random.uniform(-1,1))
            self.bias.append(np.transpose(np.matrix(temp)))

    