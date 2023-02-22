import torch
from libs.samplers import OrderedSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt


def lossSortTrainLoader(
  train_loader,
  network,
  batch_size_train: int
  ):
  with torch.no_grad():
    losses = torch.zeros(len(train_loader.dataset))
    for batch_idx, (data, target) in enumerate(train_loader):
      output = network(data)
      loss = F.cross_entropy(output, target, reduction='none')
      losses[batch_idx * len(data):batch_idx * len(data) + len(data)] = loss

    sortedLosses, sortedIndices = torch.sort(losses, descending=True)
    train_loader2 = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_train, sampler=OrderedSampler(sortedIndices), shuffle=False)

  return train_loader2

def getLossGraph(
  train_loader,
  network,
  showPlot: bool = True
):
  with torch.no_grad():
    losses = torch.zeros(len(train_loader.dataset))
    for batch_idx, (data, target) in enumerate(train_loader):
      output = network(data)
      loss = F.cross_entropy(output, target, reduction='none')    
      losses[batch_idx * len(data):batch_idx * len(data) + len(data)] = loss
    
    if (showPlot):
      losses = losses.detach().numpy()
      plt.plot(losses)
      plt.show()
      plt.pause(7)
    
    return losses