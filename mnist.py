import torch
import torchvision
from helper import plot
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from importance_samplers.loss_sort import lossSortTrainLoader, getLossGraph, equalLossTrainLoader
from importance_samplers.gradient_loss import gradientLossSortTrainLoader
from api import logRun
from libs.nodes_3d import networkTo3dNodes



n_epochs = 10
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 30

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
  batch_size=batch_size_train, shuffle=False)

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

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxPool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxPool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv2_drop = nn.Dropout2d()
        self.linearSize = self.getLinearSize()
        self.fc1 = nn.Linear(self.linearSize, 50)
        self.fc2 = nn.Linear(50, 10)
        
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
      # while True:
      #   data = next(iter(train_loader))[0]
      #   target = next(iter(train_loader))[1]



      for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
          acc = self.test()
          accPlot.append(acc)
          plot(accPlot, None)

          logRun(
            [],
            [],
            accPlot,
            [],
            [],
            'mnist',
            12,
            'gradient norm',
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
        # resort train loader
        # train_loader = equalLossTrainLoader(train_loader, network, batch_size_train)
        # train_loader = lossSortTrainLoader(train_loader, network, batch_size_train)
        # train_loader = gradientLossSortTrainLoader(train_loader, network, optimizer, batch_size_train)

      

    def test(self):
      network.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
        for data, target in test_loader:
          output = network(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
      test_loss /= len(test_loader.dataset)
      test_losses.append(test_loss)
      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
      accTensor = correct / len(test_loader.dataset)
      return accTensor.item()

network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

accPlot = []
print('Starting training')
# Activate loss sort
# train_loader = lossSortTrainLoader(train_loader, network, batch_size_train)
train_loader = gradientLossSortTrainLoader(train_loader, network, optimizer, batch_size_train)
# train_loader = equalLossTrainLoader(train_loader, network, batch_size_train)

# logNetwork(
#   batch_size_train,
#   batch_size_test,
#   'mnist',
#   learning_rate,
#   'adam',
#   'cross entropy',
#   'foobar',
# )

for epoch in range(1, n_epochs + 1):
  network.trainEpoch(epoch)

# Perform torch summery


# sum = summary(network, input_size=(3, 96, 96),device='cpu')


# print(network)

# get the network node dimensions and save them to a file

# nodes = networkTo3dNodes(network, HEIGHT, WIDTH, CHANNELS)
# log3DNodes(nodes, 'mnist')