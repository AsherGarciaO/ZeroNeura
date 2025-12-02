import src.zeroneura.utils.Images as im
import unittest

class TestMLP(unittest.TestCase):
    def testMLP(self):
        inputPath = "./tests/InputsCNN/FresaT.png"
        outputPath = "./tests/InputsCNN/FresaTR.png"

        im.resize(inputPath, [150, 150], outputPath)

if __name__ == "__main__":
    unittest.main()