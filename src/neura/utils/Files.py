import ast

def getMLPDATA(): 
    return {"LayersCount": None, "Structure": None, "LearningRate": None, "ActivationFunction": None, "Weights": None, "Biases": None}

def getCNNDATA(): 
    return {"Kernels": None, "LayersCount": None, "Structure": None, "LearningRate": None, "ActivationFunction": None, "Weights": None, "Biases": None}

def loadMLPDataNND(inputName = 'NeuralNetworkData', inputPath = './'):
    DATA = getMLPDATA()

    with open(f"{inputPath}{inputName}.nnd", "r", encoding="utf-8") as file:
        content = file.read()

    parts = content.split("Weights:")
    head = parts[0].strip()
    wieghtsAndBiases = parts[1].split("Biases:")
    weights = wieghtsAndBiases[0].strip()
    biases = wieghtsAndBiases[1].strip()

    DATA["Weights"] = ast.literal_eval(weights)
    DATA["Biases"] = ast.literal_eval(biases)

    for line in head.splitlines():
        if line.startswith("LayersCount:"):
            DATA["LayersCount"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Structure:"):
            DATA["Structure"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("ActivationFunction:"):
            DATA["ActivationFunction"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("LearningRate:"):
            DATA["LearningRate"] = float(line.split(":", 1)[1].strip())

    return DATA

def saveMLPDataNND(MLPDATA, outputName = 'NeuralNetworkData', outputPath = './'):
    with open(f"{outputPath}{outputName}.nnd", "w") as file:
        file.write("LayersCount: "+str(MLPDATA["LayersCount"])+"\n")
        file.write("Structure: "+str(MLPDATA["Structure"])+"\n")
        file.write("ActivationFunction: "+str(MLPDATA["ActivationFunction"])+"\n")
        file.write("LearningRate: "+str(MLPDATA["LearningRate"])+"\n")
        file.write("Weights: \n"+__formatTensor(MLPDATA["Weights"])+"\n")
        file.write("Biases: \n"+__formatTensor(MLPDATA["Biases"]))
        
    return True

def loadCNNDataNND(inputName = 'NeuralNetworkData', inputPath = './'):
    DATA = getCNNDATA()

    with open(f"{inputPath}{inputName}.nnd", "r", encoding="utf-8") as file:
        content = file.read()

    parts = content.split("Weights:")
    head = parts[0].strip()
    wieghtsAndBiases = parts[1].split("Biases:")
    weights = wieghtsAndBiases[0].strip()
    biasesAndKernels = wieghtsAndBiases[1].split("Kernels:")
    biases = biasesAndKernels[0].strip()
    kernels = biasesAndKernels[1].strip()

    DATA["Weights"] = ast.literal_eval(weights)
    DATA["Biases"] = ast.literal_eval(biases)
    DATA["Kernels"] = ast.literal_eval(kernels)

    for line in head.splitlines():
        if line.startswith("LayersCount:"):
            DATA["LayersCount"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Structure:"):
            DATA["Structure"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("ActivationFunction:"):
            DATA["ActivationFunction"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("LearningRate:"):
            DATA["LearningRate"] = float(line.split(":", 1)[1].strip())

    return DATA

def saveCNNDataNND(CNNDATA, outputName = 'NeuralNetworkData', outputPath = './'):
    with open(f"{outputPath}{outputName}.nnd", "w") as file:
        file.write("LayersCount: "+str(CNNDATA["LayersCount"])+"\n")
        file.write("Structure: "+str(CNNDATA["Structure"])+"\n")
        file.write("ActivationFunction: "+str(CNNDATA["ActivationFunction"])+"\n")
        file.write("LearningRate: "+str(CNNDATA["LearningRate"])+"\n")
        file.write("Weights:\n"+__formatTensor(CNNDATA["Weights"])+"\n")
        file.write("Biases:\n"+__formatTensor(CNNDATA["Biases"])+"\n")
        file.write("Kernels:\n"+__formatTensor(CNNDATA["Kernels"]))
        
    return True

def __formatTensor(tensor, indent=1, tab="\t"):
    formatted = "[\n"
    for layer in tensor:
        if isinstance(layer[0], (list, tuple)):
            formatted += tab * indent + "[\n"
            for row in layer:
                formatted += tab * (indent + 1)
                formatted += "[" + ", \t".join(f"{v}" for v in row) + "], \n"

            formatted = formatted.rstrip(", \n") + "\n"
            formatted += tab * indent + "], \n"
        else:
            formatted += tab * indent
            formatted += "[" + ", \t".join(f"{v}" for v in layer) + "], \n"

    formatted = formatted.rstrip(", \n") + "\n]"
    return formatted