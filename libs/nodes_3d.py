import torch
import torch.nn as nn

NodeTypes = {
  'linear': 'linear',
  'convolution': 'convolution',
  'pooling': 'pooling'
}

lastX = 0
lastY = 0
lastChannels = 0

def maxPoolToNode(node: nn.MaxPool2d):
  HEIGHT = 1
  h_out = (HEIGHT + 2 * node.padding - node.dilation * (node.kernel_size - 1) - 1) / node.stride + 1
  scale = HEIGHT / h_out

  global lastX
  global lastY
  x = lastX
  y = lastY

  lastX = int(x / scale)
  lastY = int(y / scale)

  return {
    'x': x,
    'y': y,
    'scale': scale,
    'type': NodeTypes['pooling'],
  }

def convToNode(node: nn.Conv2d):
  global lastChannels
  lastChannels = node.out_channels
  return {
    'x': lastX,
    'y': lastY,
    'channels': node.out_channels,
    'type': NodeTypes['convolution'],
  }

def linearToNode(node: nn.Linear):
  print(node)
  return {
    'x': lastX * lastY,
    'type': NodeTypes['linear'],
  }

def networkTo3dNodes(network: nn.Module, imageWidth: int, imageHeight: int, imageChannels: int):
  global lastX
  global lastY
  global lastChannels
  lastX = imageWidth
  lastY = imageHeight
  lastChannels = imageChannels

  nodes = []

  for node in network.modules():
    # print(node)
    def switcher(node):
      switcher = {
        nn.Conv2d: lambda: convToNode(node),
        nn.Linear: lambda: linearToNode(node),
        nn.MaxPool2d: lambda: maxPoolToNode(node),
        # nn.Dropout2d: lambda: node.p,
        nn.ReLU: lambda: None,
      }
      return switcher.get(type(node), lambda: None)()
    
    nodeOut = switcher(node)
    if nodeOut != None:
      nodes.append(nodeOut)

  return nodes
