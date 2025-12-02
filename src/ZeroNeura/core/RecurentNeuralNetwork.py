import zeroneura.utils.ActivationFunctions as af
import random

class RNNAFDebug:
    def __init__(self, structure, learningRate = 0.1, activationFunction = af.SIGMOID):
        assert isinstance(structure, list) and len(structure) >= 3, "Provide 3+ item list for network structure"
        self.structure = structure
        self.learningRate = learningRate
        self.layersCount = len(structure)
        self.activationFunction = activationFunction
        self.lastError = 0

        self.weights = []
        self.hiddenWeights = []
        self.biases = []

        for layerIn, layerOut in zip(structure[:-1], structure[1:]):
            self.weights.append([[random.uniform(-1,1) for _ in range(layerIn)] for _ in range(layerOut)])
            self.hiddenWeights.append([[random.uniform(-1,1) for _ in range(layerOut)] for _ in range(layerOut)])
            self.biases.append([random.uniform(-1,1) for _ in range(layerOut)])

    def feedforward(self, inputSequence):
        hiddenPrev = [[0]*n for n in self.structure[1:]]
        activationsSeq = []
        zsSeq = []

        for x in inputSequence:
            activations = [x]
            zs = []

            for l in range(self.layersCount-1):
                z = [sum(w*a for w,a in zip(neuron, activations[-1])) + sum(hw*hp for hw,hp in zip(neuronHidden, hiddenPrev[l])) + b for neuron, neuronHidden, b in zip(self.weights[l], self.hiddenWeights[l], self.biases[l])]
                zs.append(z)
                a = [af.activate(val, self.activationFunction) for val in z]
                activations.append(a)
                hiddenPrev[l] = a

            activationsSeq.append(activations)
            zsSeq.append(zs)

        return activationsSeq, zsSeq

    def train(self, inputSequences, outputSequences, rounds, logs = False):
        for round in range(rounds):
            totalError = 0

            if logs:
                progress = (round+1)/rounds
                bar_len = 50
                filled_len = int(bar_len*progress)
                bar = '█' * filled_len + '-'*(bar_len - filled_len)
                print(f'\rEntrenamiento: |{bar}| {round+1}/{rounds} ({progress*100:.2f}%)', end='', flush=True)
                if progress==1: print()

            for seqIn, seqOut in zip(inputSequences, outputSequences):
                activationsSeq, zsSeq = self.feedforward(seqIn)
                deltasSeq = [[None] * (self.layersCount-1) for _ in seqIn]

                for t in reversed(range(len(seqIn))):
                    y = seqOut[t] if isinstance(seqOut[t], list) else [seqOut[t]]
                    deltasSeq[t][-1] = [(yi - ai) * af.deriv(z, self.activationFunction) for yi, ai, z in zip(y, activationsSeq[t][-1], zsSeq[t][-1])]

                    for l in range(self.layersCount-3,-1,-1):
                        layerWeights = self.weights[l+1]
                        deltaNext = deltasSeq[t][l+1]
                        sp = [af.deriv(val, self.activationFunction) for val in zsSeq[t][l]]
                        deltasSeq[t][l] = [sum(layerWeights[k][j]*deltaNext[k] for k in range(len(deltaNext)))*sp[j] for j in range(len(sp))]

                    for l in range(self.layersCount-1):
                        for i in range(len(self.weights[l])):
                            for j in range(len(self.weights[l][i])):
                                self.weights[l][i][j] += self.learningRate * deltasSeq[t][l][i] * activationsSeq[t][l][j]
                            self.biases[l][i] += self.learningRate * deltasSeq[t][l][i]

                totalError += sum((y - a)**2 for t in range(len(seqOut)) for y,a in zip(seqOut[t], activationsSeq[t][-1]))
            self.lastError = totalError

    def printData(self):
        print(f"\n{'-'*40}Data{'-'*40}\n")
        print(f'Activation Function: {["Sigmoid","ReLU","Leaky ReLU","Tanh"][self.activationFunction]}')
        print(f"Structure: {self.structure}\tLayers: {self.layersCount}")
        print(f"Last Error: {self.lastError}")
        print(f"\nWeights:\n{self.weights}")
        print(f"\nBiases:\n{self.biases}")

    def predictLambda(self, inputSequences, funcion=lambda x:x, logs=True):
        results=[]
        for seq in inputSequences:
            act = self.feedforward(seq)[0]
            sequenceResults = [[funcion(v) for v in act_step[-1]] for act_step in act]
            results.append(sequenceResults)
            if logs: print(f"{seq} => {sequenceResults}")
        return results

    def predict(self, inputSequences, logs=True):
        results=[]
        for seq in inputSequences:
            act = self.feedforward(seq)[0]
            r = [v for v in act[-1][-1]]
            results.append(r)
            if logs: print(f"{seq} => {r}")
        return results