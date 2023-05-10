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
import torchvision.transforms as transforms


from samplers.samplers import Sampler, uniform, mostLoss, leastLoss, gradientNorm
from samplers.pickers import pickCdfSamples, pickOrderedSamples, pickRandomSamples

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

sampler = Sampler()

sampler.setSampler(uniform)
sampler.setPicker(pickOrderedSamples)

# Variables to be set by the user
NETWORKNAME = 'cifar - threshold testing'
RUNNUMBER = 0
TIMELIMIT = 400
SAMPLINGTHRESHOLD = 0.42
RUNNAME = 'uniform'
STARTINGSAMPLER = uniform
IMPORTANCESAMPLER = uniform
NUMBEROFRUNS = 5
WARMUPRUNS = 0

# n_epochs = 10
batch_size_train = 1024
mini_batch_size_train = 128
batch_size_test = 128
number_of_test_batches = 20
learning_rate = 0.001
momentum = 0.5
log_interval = 300
api_interval = 300
test_interval = 50

# random_seed = 1
random_seed = torch.randint(0, 100000, (1,)).item()
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# define transforms
transform = transforms.Compose(
  [transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), 
                       (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                        shuffle=False, num_workers=0)

train_loader = trainloader
test_loader = testloader

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
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32,  kernel_size=3, device=device)
        self.pool1 = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, device=device)
        self.pool2 = nn.MaxPool2d(2, 2).to(device)
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3, device=device)
        self.pool3 = nn.MaxPool2d(2, 2).to(device)
        # batch norm
        self.bn1 = nn.BatchNorm2d(32).to(device)
        
        self.fc1 = nn.Linear(512, 64, device=device)
        self.fc2 = nn.Linear(64, 10, device=device)
        
        self.currentTrainingTime = 0
        self.batchStartTime = 0
        self.initialLoss = 0
        self.importanceSamplingToggleIndex = 0

        # Plotting
        self.lossPlot = []
        self.accPlot = []
        self.timestampPlot = []

    def forward(self, x):
      x = self.pool1(F.relu(self.conv1(x)))
      x = self.pool2(F.relu(self.conv2(x)))
      x = self.pool3(F.relu(self.conv3(x)))
      #reshape
      x = x.view(x.size(0), -1)
      
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

    def trainEpoch(self, epoch):
      network.train()

      # global train_loader
      global train_loader

      # get first data sample in enumarate order from train loader
      batch_idx = 0

      # iterate over all batches
      self.batchStartTime = time.time()
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

        # end time
        self.currentTrainingTime += time.time() - self.batchStartTime

        # TESTING DO NOT FACTOR IN TIME
        if batch_idx % test_interval == 0:
          acc, lossTest = self.test()
          self.accPlot.append(acc)
          self.lossPlot.append(lossTest)
          self.timestampPlot.append(self.currentTrainingTime)
          plot(self.accPlot, None)

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

        # start time
        self.batchStartTime = time.time()

        # check variance reduction condition
        # if self.currentTrainingTime > 14:
        #   sampler.setSampler(gradientNorm)

        sampleTime = time.time()
        [data, target, _] = sampler.sample(data, target, mini_batch_size_train, network)
        # print(target)
        # print('Sample time', time.time() - sampleTime)

        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # # end time
        # self.currentTrainingTime += time.time() - start
      
        batch_idx += 1

    def test(self):
      network.eval()
      test_loss = 0
      correct = 0
      batches = 0
      with torch.no_grad():
        for data, target in test_loader:
          data = data.to(device)
          target = target.to(device)
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