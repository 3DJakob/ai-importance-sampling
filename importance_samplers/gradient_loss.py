import torch
from libs.samplers import OrderedSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import floor
from libs.logging import printProgressBar

def perSampleLoss(network, data, target):
  output = network(data)
  target = target.unsqueeze(0)
  loss = F.cross_entropy(output, target)
  return loss

def gradient_norm(loss, model):
  grad_norm = 0
  for p in model.parameters():
    if p.grad is not None:
      grad_norm += p.grad.data.norm(2).item() ** 2
  
  grad_norm = grad_norm ** (1. / 2)
  return grad_norm

def gradient_norms(data, target, model):
  grads = torch.zeros(len(data))

  # loop data points
  for i in range(len(data)):
    loss = perSampleLoss(model, data[i], target[i])
    loss.backward()
    grad_norm = gradient_norm(loss, model)
    grads[i] = grad_norm
  
  return grads

def gradientLossSortTrainLoader(
  train_loader,
  network,
  optimizer,
  batch_size_train: int
  ):
  # get the gradient of the loss
  
  # with torch.no_grad():
  losses = torch.zeros(len(train_loader.dataset))
  grads = torch.zeros(len(train_loader.dataset))
  batchCounter = 0
  TRAIN_SIZE = len(train_loader.dataset)
  BATCH_SIZE = batch_size_train
  for batch_idx, (data, target) in enumerate(train_loader):
    printProgressBar(batchCounter*BATCH_SIZE, TRAIN_SIZE, prefix = ' Training:', suffix = 'Complete', length = 50)           
    grad_norms = gradient_norms(data, target, network)
    # print(grad_norms)
    grads[batch_idx * len(data):batch_idx * len(data) + len(data)] = grad_norms
    batchCounter += 1

  # sortedLosses, sortedIndices = torch.sort(losses, descending=True)

  # sort the grads
  sortedGrads, sortedIndices = torch.sort(grads, descending=True)
  train_loader2 = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size_train, sampler=OrderedSampler(sortedIndices), shuffle=False)



  return train_loader2