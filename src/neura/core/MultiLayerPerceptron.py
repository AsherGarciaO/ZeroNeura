import src.neura.utils.ActivationFunctions as af
import src.neura.utils.Outputs as ot
import src.neura.utils.Files as f
import random

class MLP:
    def __init__(self, structure, learningRate):
        assert isinstance(structure, list) and len(structure) >= 3, "You must provide a 3 item list in the neural network structure"

        self.structure = structure
        self.learningRate = learningRate
        self.layersCount = len(structure)
        self.weights = [self.initWeights(nOut, nIn) for nIn, nOut in zip(structure[:-1], structure[1:])]
        self.biases = [[random.uniform(-1, 1) for _ in range(n)] for n in structure[1:]]
    
    def initWeights(self, rows, cols):
        return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    
    def feedforward(self, inputVector):
        activations = [inputVector]
        zs =[]

        for i in range(self.layersCount - 1):
            z = [sum(w * a for w, a in zip(neuron, activations[-1])) + b for neuron, b in zip(self.weights[i], self.biases[i])]
            zs.append(z)
            activations.append([af.sigmoid(val) for val in z])

        return activations, zs
    
    def printData(self):
        print(f"\n{'-'*40}Data{'-'*40}\n")
        print("Activation Function: Sigmoid")
        print("Counts:")
        print(f"Inputs: {self.structure[0]}\tHidden: {[self.structure[h] for h in range(1, self.layersCount-1)]}\tOutput: {self.structure[-1]}")
        print(f"Structure: {self.structure}\tLayers: {len(self.structure)}")
        print(f"\nWeights:\n{self.weights}")
        print(f"\nBias:\n{self.biases}")
    
    def train(self, inputs, outputs, rounds, logs = False):
        assert isinstance(inputs, (list)), "Inputs must be in a list"
        assert isinstance(outputs, (list)), "Outputs must be in a list"
        assert not (hasattr(outputs[0], '__iter__') and len(outputs[0]) != self.structure[-1]), f"Each output must have {self.structure[-1]} values to match the number of output neurons"
        

        outputs = [[y] if not isinstance(y, list) else y for y in outputs]

        for round in range(rounds):
            totalError = 0

            if logs and (round+1) % 1000 == 0:
                print(f"\n{'-'*40}Round #{round+1}{'-'*40}\n")
            
            for input, output in zip(inputs, outputs):
                activations, zs = self.feedforward(input)

                deltas = [None] * (self.layersCount - 1)
                deltas[-1] = [(yI - aI) * af.sigmoidDeriv(zI) for yI, aI, zI in zip(output, activations[-1], zs[-1])]

                for l in range(self.layersCount - 3, -1, -1):
                    layerWeights = self.weights[l + 1]
                    deltaNext = deltas[l + 1]
                    z = zs[l]
                    sp = [af.sigmoidDeriv(zI) for zI in z]
                    deltas[l] = [sum(layerWeights[k][j] * deltaNext[k] for k in range(len(deltaNext))) * sp[j] for j in range(len(sp))]

                for l in range(self.layersCount - 1):
                    for i in range(len(self.weights[l])):
                        for j in range(len(self.weights[l][i])):
                            self.weights[l][i][j] += self.learningRate * deltas[l][i] * activations[l][j]
                        self.biases[l][i] += self.learningRate * deltas[l][i]

                totalError += sum((yI - aI) ** 2 for yI, aI in zip(output, activations[-1]))

                if logs and (round + 1) % 1000 == 0:
                    print(f"Value: {input} =>\n{self.predict(input, False)}")
                    print(f"Error: {totalError}\n")

    def predict(self, inputs, logs = True):
        results = []

        inputs = [[i] if not isinstance(i, list) else i for i in inputs]
        for input in inputs:
            r = [round(o) for o in (self.feedforward(input)[0][-1])]
            results.append(r)

            if logs: print(f"{input} => {r}")

        return results

    def predictLambda(self, inputs, funcion = lambda x: x, logs = True):
        results = []

        inputs = [[i] if not isinstance(i, list) else i for i in inputs]
        for input in inputs:
            r = [funcion(o) for o in (self.feedforward(input)[0][-1])]
            results.append(r)

            if logs: print(f"{input} => {r}")

        return results

class MLPAF:
    def __init__(self, structure, learningRate, activationFunction = af.SIGMOID):
        assert isinstance(structure, list) and len(structure) >= 3, "You must provide a 3 item list in the neural network structure"

        self.structure = structure
        self.learningRate = learningRate
        self.layersCount = len(structure)
        self.weights = [self.initWeights(n_out, n_in) for n_in, n_out in zip(structure[:-1], structure[1:])]
        self.biases = [[random.uniform(-1, 1) for _ in range(n)] for n in structure[1:]]
        self.activationFunction = activationFunction
        self.lastError = 0
    
    def initWeights(self, rows, cols):
        return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    
    def feedforward(self, inputVector):
        activations = [inputVector]
        zs =[]

        for i in range(self.layersCount - 1):
            z = [sum(w * a for w, a in zip(neuron, activations[-1])) + b for neuron, b in zip(self.weights[i], self.biases[i])]
            zs.append(z)           
            activations.append([af.activate(val, self.activationFunction) for val in z])    

        return activations, zs
    
    def train(self, inputs, outputs, rounds, logs = False):
        assert isinstance(inputs, (list)), "Inputs must be in a list"
        assert isinstance(outputs, (list)), "Outputs must be in a list"
        assert not (hasattr(outputs[0], '__iter__') and len(outputs[0]) != self.structure[-1]), f"Each output must have {self.structure[-1]} values to match the number of output neurons"

        outputs = [[y] if not isinstance(y, list) else y for y in outputs]

        for round in range(rounds):
            totalError = 0

            if logs and (round+1) % 1000 == 0:
                print(f"\n{'-'*40}Round #{round+1}{'-'*40}\n")
            
            for input, output in zip(inputs, outputs):
                activations, zs = self.feedforward(input)
                deltas = [None] * (self.layersCount - 1)

                deltas[-1] = [(yI - aI) * af.sigmoidDeriv(zI) for yI, aI, zI in zip(output, activations[-1], zs[-1])]

                for l in range(self.layersCount - 3, -1, -1):
                    layerWeights = self.weights[l + 1]
                    deltaNext = deltas[l + 1]
                    z = zs[l]
                    sp = [af.deriv(zI, self.activationFunction) for zI in z]

                    deltas[l] = [sum(layerWeights[k][j] * deltaNext[k] for k in range(len(deltaNext))) * sp[j] for j in range(len(sp))]

                for l in range(self.layersCount - 1):
                    for i in range(len(self.weights[l])):
                        for j in range(len(self.weights[l][i])):
                            self.weights[l][i][j] += self.learningRate * deltas[l][i] * activations[l][j]
                        self.biases[l][i] += self.learningRate * deltas[l][i]

                totalError += sum((yI - aI) ** 2 for yI, aI in zip(output, activations[-1]))

                if logs and (round + 1) % 1000 == 0:
                    print(f"Value: {input} =>\n{self.predict(input, False)}")
                    print(f"Error: {totalError}\n")
                
            self.lastError = totalError
    
    def printData(self):
        print(f"\n{'-'*40}Data{'-'*40}\n")
        print(f'Activation Function: {["Sigmoid", "ReLU", "Leaky ReLU", "tanh"][self.activationFunction]}')
        print("Counts:")
        print(f"Inputs: {self.structure[0]}\tHidden: {[self.structure[h] for h in range(1, self.layersCount-1)]}\tOutput: {self.structure[-1]}")
        print(f"Structure: {self.structure}\tLayers: {len(self.structure)}")
        print(f"Last Error: {self.lastError}")
        print(f"\nWeights:\n{self.weights}")
        print(f"\nBias:\n{self.biases}")

    def predict(self, inputs, logs = True):
        results = []

        inputs = [[i] if not isinstance(i, list) else i for i in inputs]
        for input in inputs:
            r = [self.feedforward(input)[0][-1]]
            results.append(r)

            if logs: print(f"{input} => {r}")

        return results

    def predictLambda(self, inputs, funcion = lambda x: x, logs = True):
        results = []

        inputs = [[i] if not isinstance(i, list) else i for i in inputs]
        for input in inputs:
            r = [funcion(o) for o in (self.feedforward(input)[0][-1])]
            results.append(r)

            if logs: print(f"{input} => {r}")

        return results
    
class MLPAFDebug:
    def __init__(self, structure, learningRate, activationFunction = af.SIGMOID):
        assert isinstance(structure, list) and len(structure) >= 3, "You must provide a 3 item list in the neural network structure"

        self.structure = structure
        self.learningRate = learningRate
        self.layersCount = len(structure)
        self.weights = [self.initWeights(n_out, n_in) for n_in, n_out in zip(structure[:-1], structure[1:])]
        self.biases = [[random.uniform(-1, 1) for _ in range(n)] for n in structure[1:]]
        self.activationFunction = activationFunction
        self.lastError = 0
    
    def initWeights(self, rows, cols):
        return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    
    def feedforward(self, inputVector):
        activations = [inputVector]
        zs =[]

        for i in range(self.layersCount - 1):
            z = [sum(w * a for w, a in zip(neuron, activations[-1])) + b for neuron, b in zip(self.weights[i], self.biases[i])]
            zs.append(z)
            activations.append([af.activate(val, self.activationFunction) for val in z])

        return activations, zs
    
    def train(self, inputs, outputs, rounds, logs = False):
        assert isinstance(inputs, (list)), "Inputs must be in a list"
        assert isinstance(outputs, (list)), "Outputs must be in a list"
        assert not (hasattr(outputs[0], '__iter__') and len(outputs[0]) != self.structure[-1]), f"Each output must have {self.structure[-1]} values to match the number of output neurons"

        outputs = [[y] if not isinstance(y, list) else y for y in outputs]

        for round in range(rounds):
            totalError = 0

            if logs:
                progress = (round + 1) / rounds
                bar = ot.BarLoad()
                bar.showBar(progress, 'Entrenamiento: ', f' {round+1}/{rounds} ')

                if progress == 1: print()
            
            for input, output in zip(inputs, outputs):
                activations, zs = self.feedforward(input)
                deltas = [None] * (self.layersCount - 1)
                deltas[-1] = [(yI - aI) * af.sigmoidDeriv(zI) for yI, aI, zI in zip(output, activations[-1], zs[-1])]

                for l in range(self.layersCount - 3, -1, -1):
                    layerWeights = self.weights[l + 1]
                    deltaNext = deltas[l + 1]
                    z = zs[l]
                    sp = [af.deriv(zI, self.activationFunction) for zI in z]                       
                    deltas[l] = [sum(layerWeights[k][j] * deltaNext[k] for k in range(len(deltaNext))) * sp[j] for j in range(len(sp))]

                for l in range(self.layersCount - 1):
                    for i in range(len(self.weights[l])):
                        for j in range(len(self.weights[l][i])):
                            self.weights[l][i][j] += self.learningRate * deltas[l][i] * activations[l][j]
                        self.biases[l][i] += self.learningRate * deltas[l][i]

                totalError += sum((yI - aI) ** 2 for yI, aI in zip(output, activations[-1]))
                
            self.lastError = totalError
    
    def printData(self):
        print(f"\n{'-'*40}Data{'-'*40}\n")
        print(f'Activation Function: {["Sigmoid", "ReLU", "Leaky ReLU", "tanh"][self.activationFunction]}')
        print("Counts:")
        print(f"Inputs: {self.structure[0]}\tHidden: {[self.structure[h] for h in range(1, self.layersCount-1)]}\tOutput: {self.structure[-1]}")
        print(f"Structure: {self.structure}\tLayers: {len(self.structure)}")
        print(f"Last Error: {self.lastError}")
        print(f"\nWeights:\n{self.weights}")
        print(f"\nBiases:\n{self.biases}")

    def saveData(self, outputName = 'NeuralNetworkData', outputPath = './'):
        DATA = f.getMLPDATA()
        DATA["LayersCount"] = self.layersCount
        DATA["Structure"] = self.structure
        DATA["LearningRate"] = self.learningRate
        DATA["ActivationFunction"] = self.activationFunction
        DATA["Weights"] = self.weights
        DATA["Biases"] = self.biases
        
        return f.saveMLPDataNND(DATA, outputName, outputPath)
    
    def loadData(self, inputName = 'NeuralNetworkData', inputPath = './'):
        DATA = f.loadMLPDataNND(inputName, inputPath)

        self.layersCount = DATA["LayersCount"]
        self.structure = DATA["Structure"]
        self.learningRate = DATA["LearningRate"]
        self.activationFunction = DATA["ActivationFunction"]
        self.weights = DATA["Weights"]
        self.biases = DATA["Biases"]

        return True

    def predict(self, inputs, funcion = lambda x: x, logs = True):
        results = []

        inputs = [[i] if not isinstance(i, list) else i for i in inputs]
        for input in inputs:
            r = [funcion(o) for o in (self.feedforward(input)[0][-1])]
            results.append(r)

            if logs: print(f"{input} => {r}")

        return results