import numpy as np
from ActivationType import ActivationType

class Layer(object):
    '''
    This class contains the neurons, their activation functions,
    biases, and weights for single layer in the neural network.
    '''
    def __init__(self, numNeurons, numNeuronsPrevLayer, lastLayer=False,
                 dropOut=0.2, activationType=ActivationType.SIGMOID):
        # initialize the weights and biases
        self.numNeurons = numNeurons
        self.lastLayer = lastLayer
        self.numNeuronsPrevLayer = numNeuronsPrevLayer
        self.activationFunction = activationType
        self.dropOut = dropOut
        self.delta = np.zeros((numNeurons, 1))
        self.a = np.zeros((numNeurons, 1))
        self.derivAF = np.zeros((numNeurons, 1))
        self.W = np.random.randn(numNeurons, numNeuronsPrevLayer)/np.sqrt(numNeuronsPrevLayer)
        self.b = np.zeros((numNeurons,1))
        self.WGrad = np.zeros((numNeurons, numNeuronsPrevLayer))
        self.bGrad = np.zeros((numNeurons, 1))
        self.zeroout = None  # for dropout 

    def Evaluate(self, indata):
        sum = np.dot(self.W,indata) + self.b
        if (self.activationFunction == ActivationType.NONE):
            self.a = sum 
            self.derivAF = 1
        if (self.activationFunction == ActivationType.SIGMOID):
            self.a = self.sigmoid(sum)
            self.derivAF = self.a *(1 - self.a)
        if (self.activationFunction == ActivationType.TANH):
            self.a = self.TanH(sum)
            self.derivAF = (1 - self.a * self.a)
        if (self.activationFunction == ActivationType.RELU):
            self.a = self.Relu(sum)
            self.derivAF = 1.0 * (self.a > 0)
        if (self.activationFunction == ActivationType.SOFTMAX):
            self.a = self.Softmax(sum)
            self.derivAF = None  # we do delta computations in Softmax Layer
        if (self.lastLayer == False):
            self.zeroout = np.random.binomial(1, self.dropOut, (self.numNeurons,1))/self.dropOut
            self.a = self.a * self.zeroout
            self.derivAF = self.derivAF * self.zeroout

    def linear(self, x):
        return x
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))   # np.exp makes it operate on entire array 
    
    def TanH(self, x):
        return np.tanh(x)

    def Relu(self, x):
        return np.maximum(0, x)

    def Softmax(self, x):
        ex = np.exp(x)
        return ex/ex.sum()

    def ClearWBGrads(self):   # zero out accumulation of grads and biase gradients 
        self.WGrad = np.zeros((self.numNeurons, self.numNeuronsPrevLayer))
        self.bGrad = np.zeros((self.numNeurons, 1))

