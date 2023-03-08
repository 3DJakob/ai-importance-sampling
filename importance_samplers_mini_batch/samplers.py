import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def uniform (data, target, mini_batch_size):
    # pick mini batch samples randomly
    indexes = np.random.choice(data.shape[0], mini_batch_size, replace=False)
    data = data[indexes]
    target = target[indexes]
    return data, target

def mostLoss (data, target, mini_batch_size, network):
    with torch.no_grad():
      # pick mini batch samples with most loss
      output = network(data)
      loss = F.cross_entropy(output, target, reduction='none')
      indexes = torch.sort(loss, descending=True)[1][:mini_batch_size]
      data = data[indexes]
      target = target[indexes]
      return data, target
    
def leastLoss (data, target, mini_batch_size, network):
    with torch.no_grad():
      # pick mini batch samples with least loss
      output = network(data)
      loss = F.cross_entropy(output, target, reduction='none')
      indexes = torch.sort(loss, descending=False)[1][:mini_batch_size]
      data = data[indexes]
      target = target[indexes]
      return data, target

def distributeLoss (data, target, mini_batch_size, network):
    with torch.no_grad():
      output = network(data)
      loss = F.cross_entropy(output, target, reduction='none')
      indexes = torch.sort(loss, descending=True)[1]

      # pick samples with iterval to catch full range of loss
      interval = int(indexes.shape[0] / mini_batch_size)
      
      indexes = indexes[0::interval]
      data = data[indexes]
      target = target[indexes]
      return data, target
    
