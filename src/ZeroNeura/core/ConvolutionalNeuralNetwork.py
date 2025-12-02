import src.zeroneura.utils.Tensors as ts
import src.zeroneura.utils.DataProcessing as dp
import src.zeroneura.utils.ActivationFunctions as af
import src.zeroneura.utils.Images as im
import src.zeroneura.utils.Outputs as ot
import src.zeroneura.utils.Files as f
from src.zeroneura.core.MultiLayerPerceptron import MLPAFDebug as MLP
import numpy as np, cv2

class CNN:
    def __init__(self, kernels, padding = 0, stride = 1):
        self.kernels = kernels
        self.padding = padding
        self.stride = stride
        self.mlp = MLP([100, 20, 10, 1], 0.1, af.RELU)
    
    def convolve(self, img, kernel, stride = 1):
        return dp.applyKernelWithStride(img, kernel, stride)
    
    def forward(self, img, layers):
        img = im.applyGrayScale(img) if isinstance(img, str) else img

        for layerKernel in layers:
            features = []

            for kernel in layerKernel:
                fmap = self.convolve(img, kernel, self.stride)
                fmap = ts.applyLambdaToTensor(fmap, lambda x: af.ReLU(x))
                fmap = dp.maxPooling(fmap)
                features.append(fmap)

            img = features
    
        return img
    
    def train(self, frutas):
        inputs = []
        outputs = []

        for indexfruta, fruta in enumerate(frutas):
            print(f"Fruta a Analizar: {fruta}[{indexfruta+1}]")

            features = self.forward(f"./tests/Inputs/{fruta}.png")
            for _ in range(3):
                for index, feature in enumerate(features):
                    features[index] = dp.maxPooling(feature)
            CNN.showOutputs(features, f"Convolve{fruta}", f"./tests/Outputs/{fruta}/")
            print()

            for feature in features:
                feature = dp.flatten(feature)
                feature = [min(val, 255) for val in feature]
                feature = [val/255 for val in feature]
                inputs.append(feature)
                outputs.append((indexfruta+1)/10)

        self.mlp.train(inputs, outputs, 500, True)

    def predict(self, image):
        features = self.forward(image)
        for _ in range(3):
            for index, feature in enumerate(features):
                features[index] = dp.maxPooling(feature)
        CNN.showOutputs(features, f"Convolve", f"./tests/Outputs/Test/")
        print()

        maps = []

        for feature in features:
            feature = dp.flatten(feature)
            feature = [min(val, 255) for val in feature]
            feature = [val/255 for val in feature]
            maps.append(feature)

        resultados = self.mlp.predict(maps, lambda x: round(x*10), False)
        print(resultados)

        return resultados
    
    def _showOutput(img, outputName = "Output", outputPath = "./"):
        print(f"{outputName}.png")
        cv2.imwrite(f"{outputPath if outputPath != './' else ''}{outputName}.png", np.array(dp.clip(img, 0, 255)).astype(np.uint8))

    def showOutputs(featuresMaps, prefix = "Output", outputPath = "./"):
        if isinstance(featuresMaps[0], int):
            CNN._showOutput(featuresMaps, f"{prefix}", outputPath)
        else:
            for index, features in enumerate(featuresMaps):
                CNN._showOutput(features, f"{prefix}{index}", outputPath)

class CNNDebug:
    def __init__(self, kernels = [], padding = 0, stride = 1):
        self.kernels = kernels
        self.padding = padding
        self.stride = stride

        self.structureMLP = [100, 20, 10, 1]
        self.inputLayerMLP = self.structureMLP[0]
        self.mlp = MLP(self.structureMLP, 0.1, af.SIGMOID)

    def setMLPStructure(self, structureMLP, learningRate = 0.1, activationFunciton = af.SIGMOID):
        self.structureMLP = structureMLP
        self.inputLayerMLP = structureMLP[0]
        self.mlp = MLP(structureMLP, learningRate, activationFunciton)
    
    def convolve(self, img, kernel, stride = 1):
        return dp.applyKernelWithStride(img, kernel, stride)
    
    def _forwardMultiLayer(self, img, layers):      
        for layerKernel in layers:
            features = []

            for kernel in layerKernel:
                if len(ts.getShapeTensor(img)) == 2:
                    fmap = self.convolve(img, kernel, self.stride)
                else:
                    firstConv = self.convolve(img[0], kernel)
                    fmapSum = ts.createTensor(ts.getShapeTensor(firstConv), 0, 0, lambda x: round(x))

                    for subimg in img:
                        conv = self.convolve(subimg, kernel)
                        for i in range(len(conv)):
                            for j in range(len(conv[0])):
                                fmapSum[i][j] += conv[i][j]
                                
                    num_inputs = len(img)
                    for i in range(len(fmapSum)):
                        for j in range(len(fmapSum[0])):
                            fmapSum[i][j] /= num_inputs

                    fmap = fmapSum

                fmap = ts.applyLambdaToTensor(fmap, lambda x: af.ReLU(x))
                fmap = dp.maxPooling(fmap)
                features.append(fmap)


            img = features
    
        return img
    
    def forward(self, image):
        img = im.applyGrayScale(image) if isinstance(image, str) else image

        features = self._forwardMultiLayer(img, self.kernels)
        return features
    
    def train(self, imagenes, out, rounds = 1000, logs = True, outputPath = None):
        inputs = []; outputs = []

        for imagen, output in zip(imagenes, out):
            nombre = imagen.split("/")[-1].split(".")[0]
            if logs: 
                print(f"Imagen a Analizar: {nombre}")
                print(f"Nombre {nombre}")

            sizes = 0; features = imagen
            while(isinstance(features, str) or sizes > self.inputLayerMLP):
                features = self.forward(features)
                for index, feature in enumerate(features):
                    features[index] = dp.maxPooling(feature)
                
                sizes = ts.getShapeTensor(features)  
                if logs: 
                    print("Tamaño de las Imagenes", sizes, "\tDimension: ", sizes[-1]*sizes[-2])
                sizes = sizes[-1]*sizes[-2]
          
            if logs and outputPath: 
                CNNDebug.showOutputs(features, f"Convolve{nombre}", f"{outputPath}{nombre}/")
                print("\n")

            for feature in features:
                feature = dp.flatten(feature)
                feature = im.normalizeGrayImg(feature)
                
                inputs.append(self._fixLength(feature))
                outputs.append(output)

        if logs: print("\nEntrenamiento del MLP =>")
        self.mlp.train(inputs, outputs, rounds, logs)

    def predict(self, image, outputPath = None, function = lambda x: x, logs = False):
        sizes = 0; features = image
        while(isinstance(features, str) or sizes > self.inputLayerMLP):
            features = self.forward(features)
            for index, feature in enumerate(features):
                features[index] = dp.maxPooling(feature)
            
            sizes = ts.getShapeTensor(features)  
            if logs: print("Tamaño de las Imagenes", sizes)
            sizes = sizes[-1]*sizes[-2]

        if logs and outputPath:
            CNNDebug.showOutputs(features, "Convolve", outputPath)

        maps = []
        for feature in features:
            feature = dp.flatten(feature)
            feature = im.normalizeGrayImg(feature)
            maps.append(self._fixLength(feature))
        
        resultados = self.mlp.predict(maps, function, False)
        if logs: print("Resultados: ", resultados, "=> ","\n\n")

        return resultados
    
    def _fixLength(self, vec):
        target_len = self.inputLayerMLP
        vec = [float(v) for v in vec]

        if len(vec) > target_len:
            # truncar a target_len
            return vec[:target_len]
        elif len(vec) < target_len:
            return vec + ([0.0] * (target_len - len(vec)))
        else:
            return vec

    
    def saveData(self, outputName = 'zeroneuralNetworkData', outputPath = './'):
        DATA = f.getCNNDATA()
        DATA["Kernels"] = self.kernels
        DATA["LayersCount"] = self.mlp.layersCount
        DATA["Structure"] = self.mlp.structure
        DATA["LearningRate"] = self.mlp.learningRate
        DATA["ActivationFunction"] = self.mlp.activationFunction
        DATA["Weights"] = self.mlp.weights
        DATA["Biases"] = self.mlp.biases
        
        return f.saveCNNDataNND(DATA, outputName, outputPath)

    def loadData(self, inputName = 'zeroneuralNetworkData', inputPath = './'):
        DATA = f.loadCNNDataNND(inputName, inputPath)

        self.kernels = DATA["Kernels"]

        self.mlp.layersCount = DATA["LayersCount"]
        self.mlp.structure = DATA["Structure"]
        self.mlp.learningRate = DATA["LearningRate"]
        self.mlp.activationFunction = DATA["ActivationFunction"]
        self.mlp.weights = DATA["Weights"]
        self.mlp.biases = DATA["Biases"]

        return True
    
    def _showOutput(img, outputName = "Output", outputPath = "./"):
        cv2.imwrite(f"{outputPath if outputPath != './' else ''}{outputName}.png", np.array(dp.clip(img, 0, 255)).astype(np.uint8))

    def showOutputs(featuresMaps, prefix = "Output", outputPath = "./"):
        if isinstance(featuresMaps[0], int):
            CNNDebug._showOutput(featuresMaps, f"{prefix}", outputPath)
        else:
            for index, features in enumerate(featuresMaps):
                nombre = f"{prefix}{index}"

                progress = (index + 1) / len(featuresMaps)
                barra = ot.BarLoad(10)
                barra.showBar(progress, "Imprimiendo Imagenes: ", f" {index+1}/{len(featuresMaps)} {nombre} ")
                CNNDebug._showOutput(features, nombre, outputPath)