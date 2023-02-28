import torch
from libs.samplers import OrderedSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import floor


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
) -> torch.Tensor:
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

def equalLossTrainLoader(
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

    sortedLosses, sortedIndices = torch.sort(losses, descending=False)
    finalOrder = torch.zeros(len(sortedIndices), dtype=torch.long)

    # sample evenly from sortedIndices so each batch has a range of losses
    STEP_SIZE = floor(len(sortedIndices) / batch_size_train)
    for i in range(0, len(sortedIndices)):
      batchIndex = i % batch_size_train # 0-99
      batchesCompleted = floor(i / batch_size_train) # 0-599
      # finalOrder[i] = sortedIndices[batchIndex * batch_size_train + batchesCompleted]
      finalOrder[i] = sortedIndices[batchIndex * STEP_SIZE + batchesCompleted]
    # arrange losses in indice order defined by finalOrder
    lossSordedByFinalOrder = torch.zeros(len(sortedIndices))
    for i in range(0, len(sortedIndices)):
      lossSordedByFinalOrder[i] = losses[int(finalOrder[i])]

    # foo, fooindex = torch.sort(finalOrder, descending=True)

    # test = foo.detach().numpy()

    # # testHasDuplicates = False
    # # for i in range(0, len(test) - 1):
    # #   if (test[i] == test[i + 1]):
    # #     testHasDuplicates = True
    
    # # if (testHasDuplicates):
    # #   print("test has duplicates")

    # firstThousands = test[0:1000]
    # plt.plot(test)
    # plt.show()
    # plt.pause(7)


    train_loader2 = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_train, sampler=OrderedSampler(finalOrder), shuffle=False)

    # train_loader2 = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_train, sampler=OrderedSampler(sortedIndices), shuffle=False)

  return train_loader2
