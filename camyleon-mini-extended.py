import torch
import torchvision
from helper import plot
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from importance_samplers.loss_sort import lossSortTrainLoader, getLossGraph, equalLossTrainLoader
from importance_samplers.gradient_loss import gradientLossSortTrainLoader
from api import logRun, logNetwork
from libs.nodes_3d import networkTo3dNodes
from importance_samplers_mini_batch.samplers import uniform, mostLoss, distributeLoss, leastLoss, gradientNorm
import time
from libs.logging import printProgressBar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 10
batch_size_train = 1024
mini_batch_size_train = 128
batch_size_test = 1024
learning_rate = 0.0001
momentum = 0.5
log_interval = 30

# random_seed = 1
random_seed = torch.randint(0, 100000, (1,)).item()
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
import h5py

from libs.VarianceReductionCondition import VarianceReductionCondition 

reductionCondition = VarianceReductionCondition()

train_loader_camyleon = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=batch_size_train, shuffle=False)
train_loader_camyleon_targets = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=batch_size_train, shuffle=False)
# images = train_loader_camyleon.dataset['x']
test_loader_camyleon = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_test_x.h5', 'r'), batch_size=batch_size_test, shuffle=False)
test_loader_camyleon_targets = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_test_y.h5', 'r'), batch_size=batch_size_test, shuffle=False)

train_loader = train_loader_camyleon
test_loader = test_loader_camyleon

print(train_loader_camyleon_targets.dataset.keys())

# check if has .data or ['x']
trainData = None
if (hasattr(train_loader.dataset, 'data')):
  trainData = train_loader.dataset.data
else:
  trainData = train_loader.dataset['x']

if trainData is None:
  raise Exception('Could not find train data')




CHANNELS = 1
HEIGHT = trainData.shape[1]
WIDTH = trainData.shape[2]

if (len(trainData.shape) > 3):
  CHANNELS = trainData.shape[3]

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 8 * CHANNELS, kernel_size=5, device=device)
        self.maxPool = nn.MaxPool2d(2).to(device)
        self.relu = nn.ReLU().to(device)
        self.conv2 = nn.Conv2d(8 * CHANNELS, 16 * CHANNELS, kernel_size=5, device=device)
        self.maxPool2 = nn.MaxPool2d(2).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.conv2_drop = nn.Dropout2d().to(device)
        self.linearSize = self.getLinearSize()
        self.fc1 = nn.Linear(self.linearSize, 50, device=device)
        self.fc2 = nn.Linear(50, 10, device=device)
        
    def getLinearSize (self):
      testMat = torch.zeros(1, CHANNELS, HEIGHT, WIDTH, device=device)
      testMat = self.convForward(testMat)
      testMat = testMat.flatten()
      size = testMat.size().numel()
      return size

    def convForward(self, x) -> torch.Tensor:
      x = self.relu(self.maxPool(self.conv1(x)))
      # x = self.relu(self.maxPool(self.conv2_drop(self.conv2(x))))
      x = self.relu(self.maxPool(self.conv2(x)))
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

      # global train_loader
      global train_loader

      # get first data sample in enumarate order from train loader
      batch_idx = 0
      NUMBER_OF_BATCHES = int(trainData.shape[0] / batch_size_train)

      index = 0

      while index < NUMBER_OF_BATCHES:
        data = trainData[index * batch_size_train : (index + 1) * batch_size_train]
        target = train_loader_camyleon_targets.dataset['y'][index * batch_size_train : (index + 1) * batch_size_train]
        target = torch.from_numpy(target).long().squeeze().to(device)

        # data as tensor
        data = torch.from_numpy(data).float().to(device)
        data = data.permute(0, 3, 1, 2)

        # torch.Size([1024, 3, 96, 96]) data shape
        # torch.Size([1024]) target shape


      # iterate over all batches
      # for batch_idx, (data, target) in enumerate(train_loader):
        # start time
        start = time.time()

        [data, target] = uniform(data, target, mini_batch_size_train)

        # [data, target, importance] = gradientNorm(data, target, mini_batch_size_train, network)

        # importance sampling
        # [data2, target2, importance] = gradientNorm(data, target, mini_batch_size_train, network)
        # reductionCondition.update(importance)
        # if reductionCondition.satisfied.item():
        #   data = data2
        #   target = target2
        #   print('importance satisfied')
        # else:
        #   [data, target] = uniform(data, target, mini_batch_size_train)

       

        # [data, target] = mostLoss(data, target, mini_batch_size_train, network)
        # [data, target] = distributeLoss(data, target, mini_batch_size_train, network)
        # [data, target] = leastLoss(data, target, mini_batch_size_train, network)

        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
          acc = self.test()
          accPlot.append(acc)
          lossPlot.append(loss.item())
          plot(accPlot, None)

          # end time
          end = time.time()
          print('time: ' + str(end - start))

        # if batch_idx % log_interval == 0:
          logRun(
            [],
            [],
            accPlot,
            [],
            lossPlot,
            'camyleon - mini - extended',
            0,
            'uniform',
          )

          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / (train_loader.dataset['x'].shape[0] / batch_size_train), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          # torch.save(network.state_dict(), '/results/model.pth')
          # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
      
        batch_idx += 1
        # resort train loader
        # train_loader = equalLossTrainLoader(train_loader, network, batch_size_train)
        # train_loader = lossSortTrainLoader(train_loader, network, batch_size_train)
        # train_loader = gradientLossSortTrainLoader(train_loader, network, optimizer, batch_size_train)

      

    def test(self):
      network.eval()
      test_loss = 0
      correct = 0
      # NUMBER_OF_BATCHES = int(test_loader.dataset['x'].shape[0] / mini_batch_size_train)
      NUMBER_OF_BATCHES = 20
      with torch.no_grad():

        test_loader.dataset['x'].shape[0]
        batchIndex = 0


        while batchIndex < NUMBER_OF_BATCHES:
          printProgressBar(batchIndex+1, NUMBER_OF_BATCHES, length=50)

          data = test_loader.dataset['x'][batchIndex * mini_batch_size_train : (batchIndex + 1) * mini_batch_size_train]
          data = torch.from_numpy(data).float().permute(0, 3, 1, 2).to(device)

          target = test_loader_camyleon_targets.dataset['y'][batchIndex * mini_batch_size_train : (batchIndex + 1) * mini_batch_size_train]
          target = torch.from_numpy(target).long().squeeze().to(device)


        # for data, target in test_loader:
          output = network(data)
          # test_loss += F.nll_loss(output, target, reduction='sum').item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()

          batchIndex = batchIndex + 1

      numberOfSamples = NUMBER_OF_BATCHES * mini_batch_size_train

      test_loss /= numberOfSamples
      test_losses.append(test_loss)
      # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      #   test_loss, correct, numberOfSamples,
      #   100. * correct / numberOfSamples))
      accTensor = correct / numberOfSamples
      return accTensor.item()

network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

accPlot = []
lossPlot = []
print('Starting training')
# Activate loss sort
# train_loader = lossSortTrainLoader(train_loader, network, batch_size_train)
# train_loader = gradientLossSortTrainLoader(train_loader, network, optimizer, batch_size_train)
# train_loader = equalLossTrainLoader(train_loader, network, batch_size_train)

logNetwork(
  batch_size_train,
  batch_size_test,
  'camyleon - mini - extended',
  learning_rate,
  'adam',
  'cross entropy',
  'foobar',
)

for epoch in range(1, n_epochs + 1):
  network.trainEpoch(epoch)

# nodes = networkTo3dNodes(network, HEIGHT, WIDTH, CHANNELS)
# log3DNodes(nodes, 'mnist')