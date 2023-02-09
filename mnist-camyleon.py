import torch
import torchvision
from helper import plot
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.00001
momentum = 0.5
log_interval = 50

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
import h5py

train_loader_mnist = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

train_loader_pcam = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=batch_size_train, shuffle=True)
train_loader_pcam_groundTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)

test_loader_pcam_groundTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_y.h5', 'r'), batch_size=32, shuffle=True)

test_loader_mnist = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

test_loader_pcam = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_x.h5', 'r'), batch_size=batch_size_test, shuffle=True)
# Change active dataset here
train_loader = train_loader_pcam
# test_loader = test_loader_pcam
# train_loader = train_loader_mnist
test_loader = test_loader_mnist

# check if has .data or ['x']
trainData = None
if (hasattr(train_loader.dataset, 'data')):
  trainData = train_loader.dataset.data
else:
  trainData = train_loader.dataset['x']

if trainData is None:
  raise Exception('Could not find train data')


testData = test_loader_pcam.dataset['x']


CHANNELS = 1
HEIGHT = trainData.shape[1]
WIDTH = trainData.shape[2]

if (len(trainData.shape) > 3):
  CHANNELS = trainData.shape[3]

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.linearSize = self.getLinearSize()
        self.fc1 = nn.Linear(self.linearSize, 50)
        self.fc2 = nn.Linear(50, 2)
        
    def getLinearSize (self):
      testMat = torch.zeros(1, CHANNELS, HEIGHT, WIDTH)
      testMat = self.convForward(testMat)
      testMat = testMat.flatten()
      size = testMat.size().numel()
      return size

    def convForward(self, x) -> torch.Tensor:
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      return x

    def forward(self, x):
        x = self.convForward(x)
        x = x.view(-1, self.linearSize)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def trainEpoch(self, epoch):
      network.train()
      dataSize = trainData.shape[0]
      iterations = math.floor(dataSize / batch_size_train)
      print('Iterations: ' + str(iterations))
      for i in range(0, iterations):
        data = trainData[i * batch_size_train: (i + 1) * batch_size_train]
        # reshape data from 96, 96, 3 to 3, 96, 96
        data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
        data = torch.tensor(data, dtype=torch.float)
        target = train_loader_pcam_groundTruth.dataset['y'][i * batch_size_train: (i + 1) * batch_size_train]

        target = target.reshape(target.shape[0])
        target = torch.tensor(target, dtype=torch.long)

        # target2 = torch.zeros(target.shape[0], 2)
        # for i in range(0, target.shape[0]):
        #   target2[i][target[i]] = 1
        # target = target2

      # for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # correctlyClassified = 0
        # for i in range(0, output.shape[0]):
        #   if (torch.argmax(output[i]) == target[i]):
        #     correctlyClassified += 1

        # print(correctlyClassified / output.shape[0])
        # print("Loss: " + str(loss.item()))
        
        if (i+1) % log_interval == 0:
          acc = self.test()
          accPlot.append(acc)
          print(acc)
          plot(accPlot, None)

          # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          #   epoch, batch_idx * len(data), len(train_loader.dataset),
          #   100. * batch_idx / len(train_loader), loss.item()))
          # train_losses.append(loss.item())
          # train_counter.append(
          #   (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          # torch.save(network.state_dict(), '/results/model.pth')
          # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

    def test(self):
      print('Start testing')
      network.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
        dataSize = testData.shape[0]
        iterations = math.floor(dataSize / batch_size_train)
        iterations = 5 # Fast testing
        for i in range(0, iterations):
          data = testData[i * batch_size_train: (i + 1) * batch_size_train]
          # reshape data from 96, 96, 3 to 3, 96, 96
          data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
          data = torch.tensor(data, dtype=torch.float)
          target = test_loader_pcam_groundTruth.dataset['y'][i * batch_size_train: (i + 1) * batch_size_train]

          target = target.reshape(target.shape[0])
          target = torch.tensor(target, dtype=torch.long)
        # for data, target in test_loader:
          output = network(data)
          # test_loss += F.nll_loss(output, target, reduction='sum').item()
          test_loss += F.cross_entropy(output, target, reduction='sum').item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
      testSize = iterations * batch_size_train
      test_loss /= testSize
      test_losses.append(test_loss)
      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, testSize,
        100. * correct / testSize))
      accTensor = correct / testSize
      return accTensor.item()

network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

accPlot = []

for epoch in range(1, n_epochs + 1):
  network.trainEpoch(epoch)


