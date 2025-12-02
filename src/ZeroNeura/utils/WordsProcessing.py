import random, math

def tokenizeVocabulary(text):
    text = text.replace("\n", " ").replace("\t", " ").split()
    return sorted(list(set(text)))

def wordToIndex(words):
    return {word:index for index, word in enumerate(words)}

def indexToWord(words):
    return {index:word for index, word in enumerate(words)}

def oneHotWord(index, size):
    return [1 if i == index else 0 for i in range(size)]

def softmax(x, temperature = 1.0):
    x = [i/temperature for i in x]
    
    maxX = max(x)
    exp = [math.exp(i-maxX) for i in x]
    sumExp = sum(exp)

    return [j/sumExp for j in exp]

def sample(x, temperature = 1.0):
    probs = softmax(x, temperature)
    return random.choices(range(len(probs)), weights=probs)[0]