import zeroneura.utils.ActivationFunctions as af
import random

def createTensor(structure, min = 0, max = 1, function = lambda x: x):    
    if len(structure) == 1:
        return [function(random.uniform(min, max)) for _ in range(structure[0])]
    
    tensor = []
    for _ in range(structure[0]): tensor.append(createTensor(structure[1:], min, max, function))

    return tensor

def getShapeTensor(tensor):
    if isinstance(tensor, (int, float)):
        return []
    if not tensor:  # tensor vacío
        return [0]
    if isinstance(tensor[0], (int, float)):
        return [len(tensor)]
    return [len(tensor)] + getShapeTensor(tensor[0])


def applyLambdaToTensor(tensor, function = lambda x: x):
    if isinstance(tensor, (int, float)):
        return function(tensor)
    if not tensor:  # lista vacía
        return tensor
    if isinstance(tensor[0], (int, float)):
        return [function(value) for value in tensor]
    return [applyLambdaToTensor(sub, function) for sub in tensor]