import math
import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoidD(x):
    return np.multiply(sigmoid(x), sigmoid(1 - x))

class NeuralNetwork:
    learningRate = 0.1
    weightsIH = []
    weightsHO = []
    biasIH = []
    biasHO = []
    
    weights = []
    bias = []

    def __init__(self, input, hidden, output):
        self.initWeights(input, hidden, output)
        self.initBias(input, hidden, output)

    #receives and returns in list form [-,-,-]
    def guess(self, input):
        inputMatrix = np.matrix(input)
        inputMatrix = np.transpose(inputMatrix)
        
        #inputMatrix = 1 / (1 + np.exp(-inputMatrix))
        
        hiddenMatrix = np.matmul(np.transpose(self.weightsIH), inputMatrix)
        hiddenMatrix = np.add(hiddenMatrix, self.biasIH)
        
        hiddenMatrix = 1 / (1 + np.exp(-hiddenMatrix))

        outputMatrix = np.matmul(np.transpose(self.weightsHO), hiddenMatrix)
        outputMatrix = np.add(outputMatrix, self.biasHO)
        
        outputMatrix = 1 / (1 + np.exp(-outputMatrix))
        
        return outputMatrix.tolist()

    def train(self, inputSet, outputSet):
        j = 0
        for i in inputSet:
            ################################################
            outputTarget = np.matrix(outputSet[j])
            outputTarget = np.transpose(outputTarget)
            
            inputMatrix = np.matrix(i)
            inputMatrix = np.transpose(inputMatrix)
            
            #inputMatrix = 1 / (1 + np.exp(-inputMatrix))
            
            hiddenMatrix = np.matmul(np.transpose(self.weightsIH), inputMatrix)
            hiddenMatrix = np.add(hiddenMatrix, self.biasIH)
            
            hiddenMatrix = 1 / (1 + np.exp(-hiddenMatrix))

            outputMatrix = np.matmul(np.transpose(self.weightsHO), hiddenMatrix)
            outputMatrix = np.add(outputMatrix, self.biasHO)
            
            outputMatrix = 1 / (1 + np.exp(-outputMatrix))
            
            
            ##### linear algebra
            
            errorO = np.subtract(outputMatrix, outputTarget)
            
            if(random.randrange(1,1000) == 999):
                print(outputTarget, outputMatrix, errorO)
            
            errorH = np.matmul(self.weightsHO, errorO)
            
            #errorI = np.matmul(self.weightsIH, errorH)
            
            outputGradient = sigmoidD(outputMatrix)
            self.weightsHO = np.subtract(self.weightsHO, np.matmul(hiddenMatrix, np.transpose(self.learningRate * np.multiply(errorO, outputGradient))))

            hiddenGradient = sigmoidD(hiddenMatrix)
            self.weightsIH = np.subtract(self.weightsIH, np.matmul(inputMatrix, np.transpose(self.learningRate * np.multiply(errorH, hiddenGradient))))
            
            #self.weightsHO = 1 / (1 + np.exp(-self.weightsHO))
            #self.weightsIH = 1 / (1 + np.exp(-self.weightsIH))

            self.biasHO = np.subtract(self.biasHO, np.multiply(errorO, outputGradient) * self.learningRate)
            self.biasIH = np.subtract(self.biasIH, np.multiply(errorH, hiddenGradient) * self.learningRate)
            
            self.biasHO = 1 / (1 + np.exp(-self.biasHO))
            self.biasIH = 1 / (1 + np.exp(-self.biasIH))
            
            ######
            j += 1
            
    def initWeights(self, input, hidden, output):
        temp = []
        for i in range(input):
            temp2 = []
            for j in range(hidden):
                temp2.append(random.uniform(-1,1))
            temp.append(temp2)
        self.weightsIH = np.matrix(temp)
        
        #############################################
        
        temp = []
        for i in range(hidden):
            temp2 = []
            for j in range(output):
                temp2.append(random.uniform(-1,1))
            temp.append(temp2)
        self.weightsHO = np.matrix(temp)

    def initBias(self, input, hidden, output):
        temp = []
        for i in range(hidden):
            temp.append(random.uniform(-1,1))
        self.biasIH = np.transpose(np.matrix(temp))
        
        temp = []
        for i in range(output):
            temp.append(random.uniform(-1,1))
        self.biasHO = np.transpose(np.matrix(temp))

    