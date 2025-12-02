from src.ZeroNeura.core.ConvolutionalNeuralNetwork import CNNDebug as CNN
import src.ZeroNeura.utils.DataProcessing as dp
import unittest

kernel1 = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

kernel2 = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

kernel3 = [
    [0,  1,  2],
    [-1, 0,  1],
    [-2, -1, 0]
]

kernel4 = [
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2]
]

kernel5 = [
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
]

kernel6 = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
]

kernel7 = [
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1]
]

kernel8 = [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
]

kernel9 = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
]

kernel10 = [
    [0, 0, 0],
    [-1, 2, -1],
    [0, 0, 0]
]

kernels = [kernel1, kernel2, kernel3, kernel4, kernel5,kernel6, kernel7, kernel8, kernel9, kernel10]
frutas = ['Sandia', 'Manzana', 'Naranja', 'Fresa', 'FresaTR']
frutasI = [f"./tests/Inputs/CNN/{fruta}.png" for fruta in frutas]
frutasO = [1, 2, 3, 4, 4]
frutasO = [val/dp.getHigherTenMultiply(max(frutasO)) for val in frutasO]
inputPath = './tests/Outputs/CNN/'
inputName = 'TestCNN'

class TestCNN(unittest.TestCase):          
    def testCNNComplete(self):
        self.testCNNSave()
        self.testCNNLoad()

    def testCNNSave(self):        
        print("\n=== ENTRENANDO CNN ===\n")
        cnn = CNN([kernels], 5, 1)
        cnn.setMLPStructure([100, 64, 32, 1], learningRate=0.1)
        cnn.train(frutasI, frutasO, 500, True, inputPath)

        print("\n=== GUARDANDO DATOS ===\n")
        cnn.saveData(inputName, inputPath)

    def testCNNLoad(self):
        print("\n=== CARGANDO CNN Y PREDICIENDO ===\n")
        cnn1 = CNN()
        cnn1.loadData(inputName, inputPath)

        resultados = []
        for fruta in frutasI:
            print(f"\n>>> Predicción: {fruta}")
            res = cnn1.predict(fruta, inputPath + "/Test/", lambda x: round(x*10), True)
            resultados.append(res)

        print("\n=== RESULTADOS FINALES ===\n")
        for fruta, resultado, esperado in zip(frutas, resultados, frutasO):
            esperado *= 10
            procesado = dp.flatten(resultado)
            promedio = sum(procesado) / len(procesado)
            aciertos = sum(1 for x in procesado if round(x) == esperado)
            porcentaje_certeza = (aciertos / len(procesado)) * 100 if len(procesado) > 0 else 0

            print(f"Fruta: {fruta}")
            print(f"Esperado: {esperado}")
            print(f"Obtenido: {resultado}")
            print(f"\tPromedio: {promedio:.2f}")
            print(f"\tComparación: {round(promedio)} == {esperado} → {round(promedio) == esperado}")
            print(f"\tPorcentaje de Certeza: {porcentaje_certeza:.2f}%\n")

if __name__ == "__main__":
    unittest.main()