import numpy as np
import torch

def pickCdfSamples(importances, num_samples):
    importances = importances.cpu().numpy()
    # Compute the CDF of the importances
    cdf = np.cumsum(importances) / np.sum(importances)
    # Sample indices using the CDF
    indices = np.searchsorted(cdf, np.random.rand(num_samples))
    # Convert to PyTorch tensor
    indices = torch.tensor(indices, dtype=torch.long)
  
    return indices

def pickOrderedSamples (importances, mini_batch_size):
   return torch.argsort(importances, descending=True)[:mini_batch_size]

def pickSpaceBetweenSamples (indexes, mini_batch_size):
    # pick samples with equal spacing between them
    # e.g. if mini_batch_size = 4, then pick samples 0, 5, 10, 15
    return indexes[::int(len(indexes)/mini_batch_size)]

def pickRandomSamples (indexes, mini_batch_size):
    # pick random samples
    return torch.randperm(len(indexes))[:mini_batch_size]

# def mostLossEqualClasses (data, target, network):
#     with torch.no_grad():
#       # pick mini batch samples with most loss
#       output = network(data)
#       loss = F.cross_entropy(output, target, reduction='none')
#       indexes = torch.sort(loss, descending=True)[1]

#       numberOfSamplesPerClass = int(mini_batch_size / 2)
#       picked0 = 0
#       # picked1 = pickedTotal - picked0
#       pickedTotal = 0
#       i = 0

#       equalIndexes = []

#       while True:
#         if pickedTotal == mini_batch_size:
#           break

#         picked1 = pickedTotal - picked0
#         next = target[indexes[i]]

#         if next == 0 and picked0 <= numberOfSamplesPerClass:
#           picked0 += 1
#           pickedTotal += 1
#           equalIndexes.append(indexes[i])

#         if next == 1 and picked1 <= numberOfSamplesPerClass:
#           pickedTotal += 1
#           equalIndexes.append(indexes[i])

#         i += 1

#       data = data[equalIndexes]
#       target = target[equalIndexes]
#       return data, target
    