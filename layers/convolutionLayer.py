from layers.layer import layer
from scipy import signal
import activationFunctions
import numpy as np

class kernelLayer(layer):
    def __init__(self, inputShape, kernelSize, kernelDepth, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.inputShape = inputShape #3 dimensions
        self.kernelSize = kernelSize
        self.kernelDepth = kernelDepth
        inputDepth, inputHeight, inputWidth = inputShape
        self.inputDepth = inputDepth
        self.outputShape = (kernelDepth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1)
        self.kernelShape = (kernelDepth, inputDepth, kernelSize, kernelSize)
        self.initWeights()
        self.initBias()
        
        self.actFunc = activation
        self.actFuncDerivative = activationD
        
    def reinit(self) -> None:
        self.initWeights()
        self.initBias()
    
    def backPropagation(self, input, output, prevError):
        weightGradient = np.zeros(self.kernelShape)
        error = np.zeros(self.inputShape)
        gradient = self.actFuncDerivative(output) * prevError
        for i in range(self.kernelDepth):
            for j in range(self.inputDepth):
                weightGradient[i,j] = signal.correlate2d(input[j], gradient[i], "valid")
                error[j] += signal.convolve2d(gradient[i], self.weights[i,j], "full")
        self.weights -= self.learningRate * weightGradient
        self.bias -= self.learningRate * gradient
        
        return error
    
    def forward(self, input):
        output = np.copy(self.bias)
        for i in range(self.kernelDepth):
            for j in range(self.inputDepth):
                output += signal.correlate2d(input[j], self.weights[i, j], "valid")
                
        output = self.actFunc(output)
        return output

    def initWeights(self):
        self.weights = np.random.randn(*self.kernelShape)
            
    def initBias(self):
        self.bias = np.random.randn(*self.outputShape)