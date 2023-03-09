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


def compute_grad(sample, target, model, loss_fn):
    
  sample = sample.unsqueeze(0)  # prepend batch dimension for processing
  target = target.unsqueeze(0)

  prediction = model(sample)
  loss = loss_fn(prediction, target)

  # return torch.autograd.grad(loss, list(model.parameters()))
  # return torch.autograd.grad(loss, model.parameters())
  # return sum float
  return torch.autograd.grad(loss, model.parameters(), create_graph=True)[0].norm(2).item()

def gradientNorm (data, target, mini_batch_size, network):
  grads = [compute_grad(data[i], target[i], network, F.cross_entropy) for i in range(mini_batch_size)]
  # list to torch tensor
  grads = torch.FloatTensor(grads)
  # sample_grads = zip(*sample_grads)
  # sample_grads = [torch.stack(shards) for shards in sample_grads]
  # sort data by the grads
  sortedGrads, sortedIndices = torch.sort(grads, descending=True)[:mini_batch_size]
  data = data[sortedIndices]
  target = target[sortedIndices]
  return data, target

