import torch
import matplotlib.pyplot as plt

class UsageLogger:
  def __init__(self, datasetSize):
    self.datasetSize = datasetSize
    # create tensor for logging with size of dataset
    self.data = torch.zeros(datasetSize)
  
  def log(self, picked, all):
    # add one to indexes in data tensor
    self.data[picked] += 2

    # subtract one from all other indexes in data tensor
    self.data[all] -= 1
  
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
    intData = self.data.int()

    min = intData.min().item()
    max = intData.max().item()

    for i in range(min, max):
      print(i, 'count ', torch.sum(intData == i))

    print('Important sample amount ', self.data[importantIndexes])
    print('Unimportant sample amount ', self.data[unimportantIndexes])

    # save amount of samples to png
    for i in range(0, amount):
      image = dataset[importantIndexes[i]]
      imageunimportant = dataset[unimportantIndexes[i]]

      if (isinstance(image, tuple)):
        image = image[0]
        imageunimportant = imageunimportant[0]
    
      # check if image is grayscale or rgb
      if (image.shape[0] == 3):
        image = image.permute(1, 2, 0).numpy()
        imageunimportant = imageunimportant.permute(1, 2, 0).numpy()
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
        imageunimportant = ((imageunimportant - imageunimportant.min()) / (imageunimportant.max() - imageunimportant.min()) * 255).astype('uint8')
        plt.imsave('important-samples/iimage' + str(i) + '.png', image)
        plt.imsave('important-samples/uimage' + str(i) + '.png', imageunimportant)
      else:
        plt.imsave('important-samples/iimage' + str(i) + '.png', image, cmap='gray')
        plt.imsave('important-samples/uimage' + str(i) + '.png', imageunimportant, cmap='gray')

