import src.ZeroNeura.utils.Tensors as ts
import cv2

def maxPooling(tensor, size = 2, stride = 2):
    height = len(tensor)
    width = len(tensor[0])
    pooled = []

    for i in range(0, height - size + 1, stride):
        row = []
        for j in range(0, width - size + 1, stride):
            submath = [tensor[i + dy][j + dx] for dy in range(size) for dx in range(size)]
            row.append(max(submath))
        pooled.append(row)        

    return pooled

def applyKernelFromPath(inputPath, kernel):
    imagen = ts.applyGrayScale(cv2.imread(inputPath).tolist())

    img = []
    alto, ancho = ts.getShapeTensor(imagen)[:2]
    for y in range(alto):
        row = []

        for x in range(ancho):
            submath = 0
            
            testWidth = x + 1 < ancho and x - 1 >= 0
            testHeight = y + 1 < alto and y - 1 >= 0
            if testWidth and testHeight:
                for i in range(len(kernel)):
                    for j in range(len(kernel)):
                        submath += kernel[i][j]*imagen[y + i - 1][x + j - 1]

            row.append(min(submath, 255))
        img.append(row)

    return img

def applyKernelWithStride(img, kernel, stride = 1):
        h, w = ts.getShapeTensor(img)[:2]
        kh, kw = ts.getShapeTensor(kernel)[:2]

        hOut, wOut = (h-kh)//stride + 1, (w-kw)//stride + 1
        output = [[0]*wOut for _ in range(hOut)]

        for y in range(0, h - kh + 1, stride):
            for x in range(0, w - kw + 1, stride):
                suma = 0

                for i in range(kh):
                    for j in range(kw):
                        suma += img[y+i][x+j] * kernel[i][j]

                output[y//stride][x//stride] = suma

        return output

def flatten(tensor):
    if isinstance(tensor, (int, float)):
        return [tensor]

    if not tensor:
        return []

    if isinstance(tensor[0], (int, float)):
        return list(tensor)

    flat = []
    for sub in tensor:
        flat += flatten(sub)
    return flat


def clip(tensor, mini, maxi):
    return ts.applyLambdaToTensor(tensor, lambda x: max(mini, min(maxi, x)))

def getHigherTenMultiply(val):
    r = val//10 + (1 if val%10 else 0)
    return r*10