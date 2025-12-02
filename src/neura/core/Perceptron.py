class Perceptron:
    def __init__(self, weight, rateLearning, bias):
        assert isinstance(weight, (list)), "Weights must be in a list"

        self.weight = weight
        self.rateLearning = rateLearning
        self.bias = bias

    def activate(self, sum):
        return 1 if sum > 0 else 0


    def train(self, inputs, outputs, rounds, logs = False):
        assert isinstance(inputs, (list)), "Inputs must be in a list"
        assert isinstance(outputs, (list)), "Outputs must be in a list"

        if len(inputs) == len(outputs):

            for round in range(rounds):
                if logs:
                    print(f"\n{'-'*40}Round #{round+1}{'-'*40}\n")

                for i in range(len(inputs)):
                    x = inputs[i]
                    expected = outputs[i]

                    gotten = self.activate(self.calculate(x))
                    error =  expected - gotten

                    if error != 0:
                        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                            for j in range(len(x)):
                                self.weight[j] += self.rateLearning * error * x[j]
                        else:
                            self.weight[0] += self.rateLearning * error * x

                        self.bias += self.rateLearning*error

                    if logs:
                        print(f"{'-'*40}Inputs{'-'*40}")
                        print(x)
                        print(f"{'-'*40}Outputs{'-'*40}")
                        print(f"Expected: {expected}\tGotten: {gotten}")
                        print(f"Weigths: {self.weight}\tError: {error}\tBias: {self.bias}\n")

            return 1
        else:
            print("You must provide the same number of inputs and outputs")
            return 0
        
    def calculate(self, x):
        sum = 0

        if len(x) == len(self.weight):
            if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                for j in range(len(x)):
                    sum += x[j] * self.weight[j]
            else:
                sum += x*self.weight[0]
            
            sum += self.bias
        
        else:
            print("You must provide the same number of values of inputs and weights")
        
        return sum
    
    def printData(self):
        print(f"{'-'*40}Data{'-'*40}")
        print(f"Weigths: {self.weight}\tBias: {self.bias}\n")


    def predict(self, inputs, logs = False):
        assert isinstance(inputs, (list)), "Inputs must be in a list"

        results = []
        for x in inputs:
            r = self.activate(self.calculate(x))
            results.append(r)

            if logs:
                print(f"Inputs: {x}\t->\tResult: {r}")

        return results