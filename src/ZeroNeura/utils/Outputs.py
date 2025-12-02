import sys

class BarLoad:
    def __init__(self, length = 50, showPercent = True, filledBar = '█', unfilledBar = '-'):
        self.length = length
        self.filledBar = filledBar
        self.unfilledBar = unfilledBar
        self.showPercent = showPercent

    def showBar(self, progress, prefix = '', sufix = ''):
        progress = max(0, min(progress, 1))
        filledLength = int(self.length * progress)
        bar = self.filledBar * filledLength + self.unfilledBar * (self.length - filledLength)
        percentText = f"({progress * 100:.2f}%)" if self.showPercent else ''

        sys.stdout.write(f"\r{prefix}|{bar}|{sufix}{percentText}")
        sys.stdout.flush()