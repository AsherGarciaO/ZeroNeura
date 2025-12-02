import src.ZeroNeura.core.MultiLayerPerceptron as mlp
import src.ZeroNeura.utils.ActivationFunctions as af
import unittest

inputs = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]

outputs = [
    [0, 0, 0, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 0],
]

inputPath = './tests/Outputs/MLP/'
inputName = 'TestMLP'

class TestMLP(unittest.TestCase):
    def testMLP(self):
        print(f"\n{'-'*40}Prueba Unitaria Para MultiLayerPreptron{'-'*40}\n")

        mlp1 = mlp.MLP([len(inputs[0]), 8, len(outputs[0])], 0.1)
        mlp1.train(inputs, outputs, 20000)
        mlp1.printData()
        print("Test: ")
        mlp1.predictLambda(inputs, lambda r: round(r))

    def testMLPAF(self):
        print(f"\n{'-'*40}Prueba Unitaria Para MultiLayerPreptronActivationFunctions{'-'*40}\n")
        
        mlp2 = mlp.MLPAF([len(inputs[0]), 8, len(outputs[0])], 0.1, af.SIGMOID)
        mlp2.train(inputs, outputs, 2000)
        mlp2.printData()
        print("Test: ")
        mlp2.predictLambda(inputs, lambda r: round(r))
        
    def testMLPAFDebugSave(self):
        print(f"\n{'-'*40}Prueba Unitaria Para MultiLayerPreptronActivationFunctionsDebug{'-'*40}\n")

        mlp3 = mlp.MLPAFDebug([len(inputs[0]), 8, len(outputs[0])], 0.1, af.RELU)
        mlp3.train(inputs, outputs, 50000, True)
        mlp3.printData()
        
        mlp3.saveData(inputName, inputPath)
        
    def testMLPAFDebugLoad(self):
        print(f"\n{'-'*40}Prueba Unitaria Para MultiLayerPreptronActivationFunctionsDebug{'-'*40}\n")

        mlp4 = mlp.MLPAFDebug([len(inputs[0]), 8, len(outputs[0])], 0.1, af.RELU)        
        mlp4.loadData(inputName, inputPath)
        mlp4.printData()
        print("Test: ")
        mlp4.predict(inputs, lambda r: round(r))

if __name__ == "__main__":
    unittest.main()