import torch
import matplotlib.pyplot as plt

class UsageLogger:
  def __init__(self, datasetSize):
    self.datasetSize = datasetSize
    # create tensor for logging with size of dataset
    self.data = torch.zeros(datasetSize)
  
  def log(self, indexes):
    # add one to indexes in data tensor
    self.data[indexes] += 1

    # subtract one from all other indexes in data tensor
    self.data[~indexes] -= 1
  
  def getImportantSampleIndexes(self):
    # get indexes of 10 most important samples
    importantIndexes = torch.argsort(self.data, descending=True)[:10]
    return importantIndexes
  
  def getUnimportantSampleIndexes(self):
    # get indexes of 10 least important samples
    importantIndexes = torch.argsort(self.data, descending=False)[:10]
    return importantIndexes
  
  def saveSamplesToPNG(self, dataset, amount = 10):
    importantIndexes = self.getImportantSampleIndexes()
    unimportantIndexes = self.getUnimportantSampleIndexes()
    # save amount of samples to png
    for i in range(0, 10):
      image = dataset[importantIndexes[i]]
      plt.imsave('important-samples/iimage' + str(i) + '.png', image, cmap='gray')
      imageunimportant = dataset[unimportantIndexes[i]]
      plt.imsave('important-samples/uimage' + str(i) + '.png', imageunimportant, cmap='gray')

