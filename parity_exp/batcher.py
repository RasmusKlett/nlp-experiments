class Batcher:
    'Splits data into mini-batches'
    def __init__(self, data, labels, batchSize):
        self.data = data
        self.labels = labels
        self.batchSize = batchSize
        self.batchStartIndex = 0
        self.batchStopIndex = 0
        self.noData = self.data.shape[0]

    def nextBatch(self):
        self.batchStartIndex = self.batchStopIndex % self.noData
        self.batchStopIndex = min(self.batchStartIndex + self.batchSize,
                                  self.noData)
        return (
            self.data[self.batchStartIndex:self.batchStopIndex],
            self.labels[self.batchStartIndex:self.batchStopIndex]
        )
