import src.ZeroNeura.utils.Tensors as ts
import unittest

class TestMLP(unittest.TestCase):
    def testMLP(self):
        tensor = ts.createTensor([5, 4, 2])
        tensor = ts.applyLambdaToTensor(tensor, lambda x: round(x))

        print("Forma del tensor => ", ts.getShapeTensor(tensor))
        print(tensor)

if __name__ == "__main__":
    unittest.main()