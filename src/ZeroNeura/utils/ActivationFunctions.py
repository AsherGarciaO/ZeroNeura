import math

SIGMOID = 0
RELU = 1
LEAKYRELU = 2
TANH = 3

def ReLU(x): return max(0, x)    
def ReLUDeriv(x): return 1 if x > 0 else 0

def leakyReLU(x, alpha = 0.01): return x if x > 0 else alpha * x
def leakyReLUDeriv(x, alpha = 0.01): return 1 if x > 0 else alpha

def tanh(x): return math.tanh(x)    
def tanhDeriv(x): return 1 - math.tanh(x) ** 2

def sigmoid(x): return 1 / (1 + math.exp(-x))
def sigmoidDeriv(x): s = sigmoid(x); return s * (1 - s)

def activate(x, activationFunction = SIGMOID):
    if activationFunction == RELU: return ReLU(x)
    if activationFunction == LEAKYRELU: return leakyReLU(x)
    if activationFunction == TANH: return tanh(x)
    return sigmoid(x)

def deriv(x, activationFunction = SIGMOID):
    if activationFunction == RELU: return ReLUDeriv(x)
    if activationFunction == LEAKYRELU: return leakyReLUDeriv(x)
    if activationFunction == TANH: return tanhDeriv(x)
    return sigmoidDeriv(x)