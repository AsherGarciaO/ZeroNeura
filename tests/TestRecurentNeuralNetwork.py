from src.zeroneura.core.RecurentNeuralNetwork import RNNAFDebug as RNN
import src.zeroneura.utils.WordsProcessing as dp
import unittest

class TestMLP(unittest.TestCase):
    def testRNN(self):
        print(f"\n{'-'*40}Prueba Unitaria Para RecurentzeroneuralNetwork{'-'*40}\n")

        text = "hello world my name is Asher me la turbo mastuba super cabron piruje. hello world my name is Asher las ballenbas this van en el mar no en mini falda. hello world my name is Asher ollaaaaa. hello world my name is Asher coño this."

        words = text.replace("\n", " ").split()
        vocab = dp.tokenizeVocabulary(text)
        wordToIndex = dp.wordToIndex(vocab)
        indexToWord = dp.indexToWord(vocab)
        vocabSize = len(vocab)

        seqLength = 3
        inputSequences = []
        outputSequences = []

        for i in range(len(words) - seqLength):
            seqIn = words[i : i+seqLength]
            seqOut = words[i+1 : i+seqLength+1]
            inputSequences.append([dp.oneHotWord(wordToIndex[w], vocabSize) for w in seqIn])
            outputSequences.append([dp.oneHotWord(wordToIndex[w], vocabSize) for w in seqOut])

        rnn = RNN([vocabSize, 50, vocabSize], 0.1)
        rnn.train(inputSequences, outputSequences, 200, True)

        def generate_text_words(model, seed_words, length=20, temperature=1.0):
            current = seed_words[:]
            result = current[:]
            for _ in range(length):
                x = [dp.oneHotWord(wordToIndex[w], vocabSize) for w in current]
                pred = model.predict([x], logs=False)[0]
                idx = dp.sample(pred, temperature)
                next_word = indexToWord[idx]
                result.append(next_word)
                current = current[1:] + [next_word]
            return ' '.join(result)

        seed = ["hello", "world", "this"]
        print(generate_text_words(rnn, seed, length=15, temperature=0.7))

if __name__ == "__main__":
    unittest.main()