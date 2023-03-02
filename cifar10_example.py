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
from importance_samplers.loss_sort import lossSortTrainLoader, getLossGraph, equalLossTrainLoader
from libs.logging import printProgressBar
from api import logRun, logNetwork, log3DNodes
from libs.nodes_3d import networkTo3dNodes
import seaborn as sns; sns.set(style='whitegrid')





# define transforms
transform = transforms.Compose(
  [transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), 
                       (0.5, 0.5, 0.5))])

BATCH_SIZE = 32
NUM_EPOCHS = 15
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0)

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
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))
device = torch.device('mps')

print('Train size: ', len(trainloader)*BATCH_SIZE)
print('Test size: ', len(testloader)*BATCH_SIZE)
print('Classes: ', len(classes))

def init_weights(m):
  if type(m) == nn.Conv2d:
    #kaiming_init
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
      nn.init.zeros_(m.bias)


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32,  kernel_size=3)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 128,kernel_size=3)
    self.pool3 = nn.MaxPool2d(2, 2)
    # batch norm
    self.bn1 = nn.BatchNorm2d(32)
    
    self.fc1 = nn.Linear(512, 64)
    self.fc2 = nn.Linear(64, 10)

  def forward(self, x):

    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))
    #reshape
    x = x.view(x.size(0), -1)
    
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

''' Network parameters'''
lr = 0.001
criterion = nn.CrossEntropyLoss(reduction='none')
net = Net()
net.to(device)

net.apply(init_weights)
optimizer = optim.Adam(net.parameters(), lr=lr)



def get_ordered_sampler(trainloader, net, criterion, BATCH_SIZE):
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
      loss_list[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE] = loss

    printProgressBar(batch_idx+1, len(trainloader), length=50)

  # sort loss_list by loss value
  sorted_loss, indices = torch.sort(loss_list, descending=True) # True = ordered sampling
  print('\nDone sorting')
  # dataiter = iter(trainloader)
  # images, labels = next(dataiter)
  # imshow(torchvision.utils.make_grid(images))
  return OrderedSampler(indices)





print('Loading data with choosen sampler')
# sampler = get_ordered_sampler(trainloader, net, criterion, BATCH_SIZE)
# global trainloader2 
# activate equal loss sorting
# trainloader2 = equalLossTrainLoader(trainloader, net, BATCH_SIZE)
trainloader2 = trainloader

# dataiter = iter(trainloader2)
# images, labels = next(dataiter)


# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))



# validation_loss =  torch.zeros(len(trainloader2.dataset))
# for batch_idx, data in enumerate(trainloader2):
#     net.eval()
#     with torch.no_grad():
#       inputs, labels = data
#       outputs = net(inputs)
#       loss = criterion(outputs, labels)
      
#       #sort the outputs by loss value
#       # losses_b, indices = torch.sort(loss, descending=True)

#       validation_loss[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE] = loss
#       # loss_list.append(loss)

#     printProgressBar(batch_idx, len(trainloader), length=50)

# imshow(torchvision.utils.make_grid(images))

# plt.plot(validation_loss)
# plt.show()  
# plt.pause(2)

def evaluate_class_performance(net, testloader, classes):
  print('Finished Training')
  PATH = './cifar_net.pth'
  torch.save(net.state_dict(), PATH)

  dataiter = iter(testloader)
  images, labels = next(dataiter)

  # print images
  # imshow(torchvision.utils.make_grid(images))
  print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))

  net = Net()
  net.load_state_dict(torch.load(PATH))


  outputs = net(images) 
  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(BATCH_SIZE)))


  # log3DNodes(networkTo3dNodes(net, 32, 32, 3), 'CIFAR10')
  
  
  

  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  
  # again no gradients needed
  net.eval()
  with torch.no_grad():
      for data in testloader:
          images, labels = data[0] , data[1] 
          images.to(device)
          labels.to(device)
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


def evaluate_accuracy(net, testloader):
  correct = 0
  total = 0
  net.eval()
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return correct, total

def train(num_epochs, trainloader):
  loss_history={"train": [],"val": []} # history of loss values in each epoch
  metric_history={"train": [],"val": []} # histroy of metric values in each epoch

  timestamps = []
  accuracyTrain = metric_history["train"]
  accuracyTest = metric_history["val"]
  lossTrain = loss_history["train"]
  lossTest = loss_history["val"]

  '''Define network parameters to database'''
  logNetwork(
      BATCH_SIZE,
      50000,
      'CIFAR10',
      lr,
      'Adam',
      'CrossEntropyLoss',
      'Custom',
    )
  
  for epoch in range(num_epochs):  # loop over the dataset multiple times
  
    print('\nEpoch',epoch+1)
    print('Training')
    running_loss = 0.0

    
    
    # train
    net.train()
    for batch_number, data in enumerate(trainloader, 0):
      
      # get the inputs; data is a list of [inputs, labels] 
      data = next(iter(trainloader))
      inputs, labels = data

      inputs = inputs.to(device)
      labels = labels.to(device)
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)

      # sort loss values
      loss, indices = torch.sort(loss, descending=True)
      # select evenly spaced indices
      indices = indices[::len(indices)//(BATCH_SIZE//4)]

      mini_batch_inputs, mini_batch_labels = (inputs[indices], labels[indices])

      mini_batch_outputs = net(mini_batch_inputs)
      
      # Calculate mini-batch loss
      loss = criterion(mini_batch_outputs, mini_batch_labels)
      # mean loss
      loss = loss.sum()
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()

      printProgressBar(batch_number+1, len(trainloader.dataset)/BATCH_SIZE, length=50)

    # print loss
    print()
    print(f'[{epoch + 1}] loss: {running_loss / len(trainloader.dataset):.3f}')

    correct, total = evaluate_accuracy(net, testloader)
    

    print()
    print(f'Accuracy on {total} test images: {correct} / {total}    = {100 * correct / total:.2f} %')

    metric_history["train"].append(100 * correct // total)
    loss_history["train"].append(running_loss / len(trainloader.dataset))

    metric_history["val"].append(100 * correct // total)
    loss_history["val"].append(running_loss / len(trainloader.dataset))
    
    '''Save data to database'''
    logRun(
        timestamps,
        accuracyTrain,
        accuracyTest,
        lossTrain,
        lossTest,
        'CIFAR10',
        1,
        'Random sampling',
      )
    # print('Updating data loader')
    # trainloader = equalLossTrainLoader(trainloader, net, BATCH_SIZE)
  return loss_history, metric_history
  



  
loss_history, metric_history = train(NUM_EPOCHS , trainloader2)

evaluate_class_performance(net, testloader, classes)

# print graph of accuracy and loss sns
fig, ax = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
sns.lineplot(x=range(len(loss_history["train"])), y=loss_history["train"], ax=ax[0], label="train")
sns.lineplot(x=range(len(loss_history["val"])), y=loss_history["val"], ax=ax[0], label="val")
ax[0].set_title("Loss")
sns.lineplot(x=range(len(metric_history["train"])), y=metric_history["train"], ax=ax[1], label="train")
sns.lineplot(x=range(len(metric_history["val"])), y=metric_history["val"], ax=ax[1], label="val")
ax[1].set_xlabel("Accuracy")

plt.suptitle("Loss and accuracy")
plt.show()