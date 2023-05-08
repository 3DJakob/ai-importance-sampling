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
from samplers.samplers import uniform, mostLoss, leastLoss, gradientNorm
import time
import h5py

from libs.VarianceReductionCondition import VarianceReductionCondition 


from samplers.samplers import Sampler, uniform, mostLoss, leastLoss, gradientNorm
from samplers.pickers import pickCdfSamples, pickOrderedSamples, pickRandomSamples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampler = Sampler()

sampler.setSampler(uniform)
sampler.setPicker(pickOrderedSamples)

# Variables to be set by the user
NETWORKNAME = 'mnist - threshold testing'
RUNNUMBER = 0
TIMELIMIT = 30
SAMPLINGTHRESHOLD = 0.10
RUNNAME = 'gradient norm %f threshold' % SAMPLINGTHRESHOLD
STARTINGSAMPLER = uniform
IMPORTANCESAMPLER = gradientNorm
NUMBEROFRUNS = 10
WARMUPRUNS = 0

# n_epochs = 10
batch_size_train = 1024
mini_batch_size_train = 128
batch_size_test = 128
number_of_test_batches = 20
learning_rate = 0.01
momentum = 0.5
log_interval = 300
api_interval = 300
test_interval = 0.2

# random_seed = 1
random_seed = torch.randint(0, 100000, (1,)).item()
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

reductionCondition = VarianceReductionCondition()

train_loader_mnist = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader_mnist = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=False)

train_loader = train_loader_mnist
test_loader = test_loader_mnist

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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, device=device)
        self.maxPool = nn.MaxPool2d(2).to(device)
        self.relu = nn.ReLU().to(device)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, device=device)
        self.maxPool2 = nn.MaxPool2d(2).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.conv2_drop = nn.Dropout2d().to(device)
        self.linearSize = self.getLinearSize()
        self.fc1 = nn.Linear(self.linearSize, 50, device=device)
        self.fc2 = nn.Linear(50, 10, device=device)
        self.currentTrainingTime = 0
        self.initialLoss = 0
        self.importanceSamplingToggleIndex = 0

        # Plotting
        self.lossPlot = []
        self.accPlot = []
        self.timestampPlot = []
        
    def getLinearSize (self):
      testMat = torch.zeros(1, CHANNELS, HEIGHT, WIDTH)
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

      # iterate over all batches
      
      for batch_idx, (data, target) in enumerate(train_loader):
        if self.currentTrainingTime > TIMELIMIT:
          print('Time limit reached', self.currentTrainingTime)
          logRun(
            self.timestampPlot,
            [],
            self.accPlot,
            [],
            self.lossPlot,
            NETWORKNAME,
            RUNNUMBER,
            RUNNAME,
            self.importanceSamplingToggleIndex
          )
          break

        # Move tensors to the GPU
        data = data.to(device)
        target = target.to(device)

        acc, lossTest = self.test()
        self.accPlot.append(acc)
        self.lossPlot.append(lossTest)
        self.timestampPlot.append(self.currentTrainingTime)
        plot(self.accPlot, None)

        # start time
        start = time.time()

        # check variance reduction condition
        # if self.currentTrainingTime > 14:
        #   sampler.setSampler(gradientNorm)

        [data, target, _] = sampler.sample(data, target, mini_batch_size_train, network)

        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # end time
        self.currentTrainingTime += time.time() - start


        if batch_idx % log_interval == 0:
          logRun(
            self.timestampPlot,
            [],
            self.accPlot,
            [],
            self.lossPlot,
            NETWORKNAME,
            RUNNUMBER,
            RUNNAME,
            self.importanceSamplingToggleIndex
          )

          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / (train_loader.dataset.data.shape[0] / batch_size_train), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          # torch.save(network.state_dict(), '/results/model.pth')
          # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
      
        batch_idx += 1

    def test(self):
      network.eval()
      test_loss = 0
      correct = 0
      batches = 0
      with torch.no_grad():
        for data, target in test_loader:
          if batches > number_of_test_batches:
            break
          output = network(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()

          batches += 1
      test_loss /= (batch_size_test * batches)
      test_losses.append(test_loss)

      if self.initialLoss == 0:
        self.initialLoss = test_loss

      if self.initialLoss * SAMPLINGTHRESHOLD > test_loss:
        print('Sampling threshold reached')
        
        if self.importanceSamplingToggleIndex == 0:
          sampler.setSampler(IMPORTANCESAMPLER)
          self.importanceSamplingToggleIndex = len(self.lossPlot)

      # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      #   test_loss, correct, (batch_size_test * batches),
      #   100. * correct / (batch_size_test * batches)))
      accTensor = correct / (batch_size_test * batches)
      return accTensor.item(), test_loss
    
    def reset(self):
      self.currentTrainingTime = 0
      # reset weights
      self.fc1.reset_parameters()
      self.fc2.reset_parameters()
      self.conv1.reset_parameters()
      self.conv2.reset_parameters()

      self.accPlot = []
      self.lossPlot = []
      self.timestampPlot = []
      self.initialLoss = 0
      self.importanceSamplingToggleIndex = 0
      sampler.setSampler(STARTINGSAMPLER)
      
      global RUNNUMBER
      global WARMUPRUNS
      global NUMBEROFRUNS

      if (WARMUPRUNS > 0):
        WARMUPRUNS = WARMUPRUNS - 1
      else:
        RUNNUMBER = RUNNUMBER + 1
        NUMBEROFRUNS = NUMBEROFRUNS - 1

network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []

print('Starting training')
# Activate loss sort
# train_loader = lossSortTrainLoader(train_loader, network, batch_size_train)
# train_loader = gradientLossSortTrainLoader(train_loader, network, optimizer, batch_size_train)
# train_loader = equalLossTrainLoader(train_loader, network, batch_size_train)

# logNetwork(
#   batch_size_train,
#   batch_size_test,
#   NETWORKNAME,
#   learning_rate,
#   'adam',
#   'cross entropy',
#   'custom',
# )


# for epoch in range(1, n_epochs + 1):
epoch = 1
while True:
  if network.currentTrainingTime > TIMELIMIT:
    network.reset()
    epoch = 1 
    print('Starting new run', RUNNUMBER)

  if NUMBEROFRUNS == 0:
    break

  network.trainEpoch(epoch)
  epoch += 1

print(network.currentTrainingTime)

# nodes = networkTo3dNodes(network, HEIGHT, WIDTH, CHANNELS)
# log3DNodes(nodes, 'mnist')