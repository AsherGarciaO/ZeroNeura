from src.neura.core.Perceptron import Perceptron
import unittest

class TestMLP(unittest.TestCase):
    def testMLP(self):
        print(f"\n{'-'*40}Prueba Unitaria Para Preptron{'-'*40}\n")
        inputs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

        outputs = [0, 0, 0, 1]

        perceptron = Perceptron([0.0, 0.0], 0.1, 0)
        perceptron.train(inputs, outputs, 100)
        perceptron.predict([[0, 0], [1, 1], [0, 1]], True)

if __name__ == "__main__":
    unittest.main()