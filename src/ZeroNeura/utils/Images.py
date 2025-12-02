import zeroneura.utils.Tensors as ts
import zeroneura.utils.DataProcessing as dp
import cv2, numpy as np

def resize(inputPath, newSize, outputPath):
    img = cv2.imread(inputPath)
    resized = cv2.resize(img, newSize)

    cv2.imwrite(outputPath, resized)

def applyGrayScale(inputValue):
    img = cv2.imread(inputValue).tolist()
    alto, ancho = ts.getShapeTensor(img)[:2]

    for y in range(alto):
        for x in range(ancho):
            r, g, b = img[y][x]
            img[y][x] = int(0.299*r + 0.587*g + 0.114*b)
    
    return img

def normalizeGrayImg(img, maxValue = None):
    img = ts.applyLambdaToTensor(img, lambda x: max(0, min(255, x)))

    maxValue = max(img) if maxValue == None else maxValue
    maxValue = dp.getHigherTenMultiply(maxValue)
    img = ts.applyLambdaToTensor(img, lambda x: x/maxValue)

    return img
    
def applyPadding(img, padding = 0, value = 0):
    h, w = ts.getShapeTensor(img)[:2]
    padded = [[value]* (w + 2*padding) for _ in range(h + 2*padding)]
    
    for i in range(h):
        for j in range(w):
            padded[i + padding][j + padding] = img[i][j]

    return padded    

def showOutput(img, outputName = "Output", outputPath = "./"):
    cv2.imwrite(f"{outputPath if outputPath != './' else ''}{outputName}.png", np.array(dp.clip(img, 0, 255)).astype(np.uint8))