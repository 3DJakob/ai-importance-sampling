import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from libs.samplers import OrderedSampler
from libs.logging import printProgressBar
from api import logRun, logNetwork, log3DNodes
from libs.nodes_3d import networkTo3dNodes



if __name__ == '__main__':
  # define transforms
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  batch_size = 16
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # functions to show an image
  def imshow(img):
      img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()

  # get some random training images
  dataiter = iter(trainloader)
  images, labels = next(dataiter)

  # get image size
  print(images[0].shape)

  # show images
  # imshow(torchvision.utils.make_grid(images))
  # # print labels
  # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
  device = torch.device('mps')

  print('Train size: ', len(trainloader)*batch_size)
  print('Test size: ', len(testloader)*batch_size)
  print('Classes: ', len(classes))

  class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 6,  kernel_size=5)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.conv3 = nn.Conv2d(16, 32,kernel_size=3)
      self.pool3 = nn.MaxPool2d(2, 2)
      
      self.fc1 = nn.Linear(128, 64)
      self.fc2 = nn.Linear(64, 32)
      self.fc3 = nn.Linear(32, 16)
      self.fc4 = nn.Linear(16, 10)

    def forward(self, x):

      x = self.pool1(F.relu(self.conv1(x)))
      x = self.pool2(F.relu(self.conv2(x)))
      x = self.pool3(F.relu(self.conv3(x)))
      x = torch.flatten(x, 1) # Flatten all dimensions except batch
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

  lr = 0.003
  net = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
  
  loss_list = torch.zeros(len(trainloader.dataset))
  print('Sorting loss list')
  for batch_idx, data in enumerate(trainloader):
    net.eval()
    with torch.no_grad():
      inputs, labels = data
      inputs = inputs
      labels = labels

      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss_list[batch_idx*batch_size:batch_idx*batch_size+batch_size] = loss

    printProgressBar(batch_idx+1, len(trainloader), length=50)
  
  # sort loss_list by loss value
  sorted_loss, indices = torch.sort(loss_list, descending=True) # True = ordered sampling
  print('\nDone sorting')
  # dataiter = iter(trainloader)
  # images, labels = next(dataiter)
  # imshow(torchvision.utils.make_grid(images))

  

  '''Ordered Sampler'''
  batch_size = 16
  sampler = OrderedSampler(indices)

  print('Loading data with ordered sampler')
  trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            num_workers=2, shuffle=True)
  testloader2 = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           num_workers=2, shuffle=False)

  # dataiter = iter(trainloader2)
  # images, labels = next(dataiter)


  # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



  # validation_loss =  torch.zeros(len(trainloader2.dataset))
  # for batch_idx, data in enumerate(trainloader2):
  #     net.eval()
  #     with torch.no_grad():
  #       inputs, labels = data
  #       outputs = net(inputs)
  #       loss = criterion(outputs, labels)
        
  #       #sort the outputs by loss value
  #       # losses_b, indices = torch.sort(loss, descending=True)

  #       validation_loss[batch_idx*batch_size:batch_idx*batch_size+batch_size] = loss
  #       # loss_list.append(loss)

  #     printProgressBar(batch_idx, len(trainloader), length=50)

  # imshow(torchvision.utils.make_grid(images))

  # plt.plot(validation_loss)
  # plt.show()  
  # plt.pause(2)
  criterion2 = nn.CrossEntropyLoss()

  # print length of tensor trainloader2
  print(len(trainloader2.dataset))

  loss_history={"train": [],"val": []} # history of loss values in each epoch
  metric_history={"train": [],"val": []} # histroy of metric values in each epoch
  
  timestamps = []
  accuracyTrain = metric_history["train"]
  accuracyTest = metric_history["val"]
  lossTrain = loss_history["train"]
  lossTest = loss_history["val"]

  logNetwork(
      batch_size,
      len(trainloader2.dataset),
      'CIFAR10',
      lr,
      'SGD',
      'CrossEntropyLoss',
      'foobar',
    )

  for epoch in range(10):  # loop over the dataset multiple times
  
    print('\nEpoch',epoch+1)
    print('Training')
    running_loss = 0.0
    # train
    net.train()
    for batch_idx, data in enumerate(trainloader2, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        inputs = inputs
        labels = labels

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion2(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        printProgressBar(batch_idx+1, len(trainloader2.dataset)/batch_size, length=50)
    # print loss
    print()
    print(f'[{epoch + 1}] loss: {running_loss / len(trainloader2.dataset):.3f}')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    net.eval()
    with torch.no_grad():
      print('Testing')
      counter = 1
      for data in testloader2:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images) 
        # the class with the highest energy is what we choose as prediction
 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        printProgressBar(counter, len(testloader2.dataset)/batch_size, length=50)
        counter += 1

    print()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    metric_history["train"].append(100 * correct // total)
    loss_history["train"].append(running_loss / len(trainloader2.dataset))

    metric_history["val"].append(100 * correct // total)
    loss_history["val"].append(running_loss / len(trainloader2.dataset))
    
    logRun(
        timestamps,
        accuracyTrain,
        accuracyTest,
        lossTrain,
        lossTest,
        'CIFAR10',
        8,
        'Random sampling',
      )

  print('Finished Training')
  PATH = './cifar_net.pth'
  torch.save(net.state_dict(), PATH)

  dataiter = iter(testloader2)
  images, labels = next(dataiter)

  # print images
  imshow(torchvision.utils.make_grid(images))
  print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

  net = Net()
  net.load_state_dict(torch.load(PATH))


  outputs = net(images) 
  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(batch_size)))


  # log3DNodes(networkTo3dNodes(net, 32, 32, 3), 'CIFAR10')
  
  
  
  # correct = 0
  # total = 0
  # # since we're not training, we don't need to calculate the gradients for our outputs
  # net.eval()
  # with torch.no_grad():
  #   for data in testloader2:
  #     images, labels = data
  #     # calculate outputs by running images through the network
  #     outputs = net(images) 
  #     # the class with the highest energy is what we choose as prediction
  #     _, predicted = torch.max(outputs.data, 1)
  #     total += labels.size(0)
  #     correct += (predicted == labels).sum().item()

  # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  net.eval()
  with torch.no_grad():
      for data in testloader2:
          images, labels = data[0] , data[1] 
          outputs = net(images) 
          _, predictions = torch.max(outputs, 1)
          # collect the correct predictions for each class
          for label, prediction in zip(labels, predictions):
              if label == prediction:
                  correct_pred[classes[label]] += 1
              total_pred[classes[label]] += 1


  # print accuracy for each class
  for classname, correct_count in correct_pred.items():
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')