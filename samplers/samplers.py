import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from samplers.pickers import pickCdfSamples, pickOrderedSamples

class Sampler:
    def __init__(self):
      self.picker = pickCdfSamples
      self.sampler = uniform

    def sample(self, data, target, mini_batch_size, network):
      importance = self.sampler(data, target, network).cpu()

      # indexes for sorting by importance
      importanceIndexes = torch.argsort(importance, descending=True)
      pickerIndexes = self.picker(importance[importanceIndexes], mini_batch_size)
      
      target = target[importanceIndexes][pickerIndexes]
      data = data[importanceIndexes][pickerIndexes]

      # check max of pickerIndexes larger than data.shape[0]
      if pickerIndexes.max().item() >= importance.shape[0]:
        print(pickerIndexes.max().item(), 'pickerIndexes.max() >= data.shape[0]', importance.shape[0])
        # print(pickerIndexes)


      if importanceIndexes.max().item() >= importance.shape[0]:
        print(importanceIndexes.max().item(), 'importanceIndexes.max() >= data.shape[0]', importance.shape[0])
        # print(importanceIndexes)

      return data, target, importance[importanceIndexes][pickerIndexes], importanceIndexes[pickerIndexes]
    
    def setSampler (self, sampler):
      self.sampler = sampler
    
    def setPicker (self, picker):
      self.picker = picker

### SAMPLERS ###
# All samplers have the same signature:
# def sampler (data, target, network):
#   return importance

def uniform (data, target, network):
    # pick mini batch samples randomly
    # indexes = np.random.choice(data.shape[0], data.shape[0], replace=False)
    importance = torch.rand(data.shape[0])
    return importance

def mostLoss (data, target, network):
    with torch.no_grad():
      # pick mini batch samples with most loss
      output = network(data)
      loss = F.cross_entropy(output, target, reduction='none')
      return loss

def leastLoss (data, target, network):
    with torch.no_grad():
      # pick mini batch samples with least loss
      output = network(data)
      loss = F.cross_entropy(output, target, reduction='none')
      importance = -loss
      return importance

def compute_grad(sample, target, model, loss_fn, use_last_layer_grad=False):
    
  sample = sample.unsqueeze(0)  # prepend batch dimension for processing
  target = target.unsqueeze(0)

  prediction = model(sample)
  loss = loss_fn(prediction, target)

  if use_last_layer_grad:
    lastLayer = list(model.parameters())[-1]
    lastLayerGrad = torch.autograd.grad(loss, lastLayer, create_graph=True)[0].norm(2).item()
    return lastLayerGrad

  return torch.autograd.grad(loss, model.parameters(), create_graph=True)[0].norm(2).item()

def gradientNorm (data, target, network):
  grads = [compute_grad(data[i], target[i], network, F.cross_entropy, True) for i in range(data.shape[0])]
  # list to torch tensor
  importance = torch.FloatTensor(grads)
  return importance
