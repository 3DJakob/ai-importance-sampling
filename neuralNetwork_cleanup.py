import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import h5py
from torchvision.transforms import transforms
import random
from helperPlot import plot
from display import *
from torch import Tensor




BATCH_SIZE = 32
TRAIN_SIZE = 3200
TEST_SIZE = 800
number_of_labels = 2

# Create an instance for training. 
train_set_data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=BATCH_SIZE, shuffle=False)
train_set_labels = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=BATCH_SIZE, shuffle=False)

# Create an instance for testing, shuffle is set to False.
test_set_data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_x.h5', 'r'), batch_size=BATCH_SIZE, shuffle=False)
test_set_labels = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_y.h5', 'r'), batch_size=BATCH_SIZE, shuffle=False)

print("The number of images in the training set: ", len(train_set_data.dataset['x']))
print("The number of images in the test set: ", len(test_set_data.dataset['x']))

print("Training size:", TRAIN_SIZE)
print("Test size:", TEST_SIZE)
print("Batch size:", BATCH_SIZE)


classes = ('cancer', 'no cancer')

 # Define your execution device
device = torch.device("mps") # if you have a GPU, change to "cuda"
print("The model will be running on", device, "device")

# initialize the network weights with He
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(12).to(device)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(24).to(device)

        self.pool = nn.MaxPool2d(2, 2).to(device)
        
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1).to(device)
        self.bn4 = nn.BatchNorm2d(48).to(device)

        self.conv5 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1).to(device)
        self.bn5 = nn.BatchNorm2d(64).to(device)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1).to(device)
        self.bn6 = nn.BatchNorm2d(128).to(device)


        self.fc1 = nn.Linear(3200, 1280).to(device)
        self.fc2 = nn.Linear(1280, 256).to(device)
        self.fc3 = nn.Linear(256, 128).to(device)
        self.fc4 = nn.Linear(128, 64).to(device)
        self.fc5 = nn.Linear(64, 32).to(device)
        self.fc6 = nn.Linear(32, 16).to(device)
        self.fc7 = nn.Linear(16, 8).to(device)
        self.fc8 = nn.Linear(8, 4).to(device)
        self.fc9 = nn.Linear(4, 2).to(device)
        # initialize the network weights
        self.apply(init_weights)

    def forward(self, x):
        # Permute the tensor to (batch_size, channels, height, width)
        x = torch.permute(x, (0, 3, 1, 2)).to(device)

        x = F.relu(self.bn1(self.conv1(x).to(device))).to(device)    
        x = F.relu(self.bn2(self.conv2(x).to(device))).to(device)
        
        x = self.pool(x).to(device)
        
        x = F.relu(self.bn4(self.conv4(x).to(device))).to(device)
        
        x = self.pool(x).to(device)

        x = F.relu(self.bn5(self.conv5(x).to(device))).to(device)   
        
        x = self.pool(x).to(device)

        x = F.relu((self.conv6(x))).to(device)

        x = self.pool(x).to(device)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1).to(device)

        x = F.relu(self.fc1(x).to(device)).to(device)
        x = F.relu(self.fc2(x).to(device)).to(device)
        x = F.relu(self.fc3(x).to(device)).to(device)
        x = F.relu(self.fc4(x).to(device)).to(device)        
        x = F.relu(self.fc5(x).to(device)).to(device)
        x = F.relu(self.fc6(x).to(device)).to(device)
        x = F.relu(self.fc7(x).to(device)).to(device)
        x = F.relu(self.fc8(x).to(device)).to(device)

        x = (self.fc9(x).to(device))
        return x

model = Network()


loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

def printProgressBar (iteration, total, length, prefix = '', suffix = '', decimals = 1, fill = '█', printEnd = "\r"):
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print new line on complete
    if iteration >= total: 
        print()


# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    correctImages = 0
    
    with torch.no_grad():
        predictedTrue = 0
        predictedFalse = 0
        for j in range(0, TEST_SIZE, BATCH_SIZE):
            images = test_set_data.dataset['x'][j:j+BATCH_SIZE]
            labels = test_set_labels.dataset['y'][j:j+BATCH_SIZE]
            # add row to labels that inverts the labels
            labels = np.column_stack((labels, np.logical_not(labels)))
            # reshape the labels to be a 2x10 matrix
            labels = labels.reshape(BATCH_SIZE, 2)

            # convert images to tensor
            images = torch.tensor(images, dtype=torch.float32).to(device)
            # normalize the images
            images = images / 255.0
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            
            # run the model on the test set to predict labels
            outputs = model(images)
            # one-hot encode the outputs
            outputs = torch.nn.functional.one_hot(torch.argmax(outputs, dim=1), num_classes=2).to(device)
            
            # sum the elements of outputs
            predictedTrue = torch.sum(outputs[:,0])
            predictedFalse = torch.sum(outputs[:,1])
            
            labelsRatio = labels.mean()
            
            # count the number of correct images with a mask where the outputs and labels are equal
            _, counts = torch.unique(outputs[outputs == labels], return_counts=True)
            if len(counts) > 0:
                correctImages += counts[0].item()
            else:
                correctImages += 0

            printProgressBar(j+BATCH_SIZE, TEST_SIZE, prefix = ' Testing:', suffix = 'Complete', length = 50)
            
        # print prediction ratio
        print("Prediction ratio: False [", end='')
        for i in range(int(predictedTrue/(predictedTrue+predictedFalse)*10)):
            print('-', end='')
        print('*', end='')
        for i in range(int(predictedFalse/(predictedTrue+predictedFalse)*10)):
            print('-', end='')
        print('] True')
        
        # print labels ratio
        print("Batch ratio: {0:4} False [".format(""), end='')
        for i in range(int(labelsRatio.item()*10)):  
            print('-', end='')
        print('*', end='')
        for i in range(int((labelsRatio.item())*10)):
            print('-', end='')
        print('] True')
        
        # compute the accuracy over all test images
        accuracy = correctImages / TEST_SIZE 
    print(correctImages, '/', TEST_SIZE, 'images correct. Accuracy %.2f %%' % (correctImages/TEST_SIZE*100))
    return(accuracy)

# Training function. 
# loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0
    accuracyPlot = []
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # Train the model
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        batchCounter = 0
        print('Epoch ', epoch+1)
        
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            i = random.randint(0, 260000)
            # get the inputs
            images = train_set_data.dataset['x'][i:i+BATCH_SIZE]
            labels = train_set_labels.dataset['y'][i:i+BATCH_SIZE]
            #add row to labels that inverts the labels
            labels = np.column_stack((labels, np.logical_not(labels)))
            # reshape the labels to be a 2x10 matrix
            labels = labels.reshape(BATCH_SIZE, 2)
           
            # convert to torch tensors with the float32 data type
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            images = torch.tensor(images, dtype=torch.float32).to(device)
            #normalize the images
            images = images / 255.0

            # zero the parameter gradients
            optimizer.zero_grad()
    
            outputs = model(images)
            #softmax the outputs
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            # compute the loss of the model
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()     # extract the loss value            
            batchCounter = batchCounter + 1
            printProgressBar(batchCounter*BATCH_SIZE, TRAIN_SIZE, prefix = ' Training:', suffix = 'Complete', length = 50)           
    
        accuracy = testAccuracy()
        # Save the model if the accuracy is the best
        if accuracy > best_accuracy:
            print('Saving the model with accuracy %f %%' % (accuracy))
            saveModel()
            best_accuracy = accuracy
        accuracyPlot.append(accuracy)
        plot(accuracyPlot, None)

        print('Loss: %.4f' %( running_loss / TRAIN_SIZE))
        print()

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show(inp, label):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    plt.imshow(inp)
    plt.title(label)
    plt.pause(5)  # pause a bit so that plots are updated


# Function to test print a batch of images
def printBatch():
    # get batch of images from the test_set
    images = test_set_data.dataset['x'][0:BATCH_SIZE]
    labels = test_set_labels.dataset['y'][0:BATCH_SIZE]

    #convert entries to images
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    #permute the images to be in the correct format
    images = images.permute(0, 3, 1, 2)
    images = images / 255.0
    print (images.shape)

    labels = labels.flatten()
    #convert to integers    
    labels = labels.type(torch.LongTensor)
    
    # show all images as one image grid
    grid = torchvision.utils.make_grid(images, nrow=25)
    show(grid, label='GroundTruth: ' + ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))
    
    #save images
    torchvision.utils.save_image(grid, 'test.png')

if __name__ == "__main__":
    # Build model with test after each epoch
    train(50) # epochs

    
    # Load the model and test the accuracy
    model = Network()
    path = "myFirstModel.pth"

    model.load_state_dict(torch.load(path))
    print('Model loaded from %s' % path)

    testAccuracy()

    # Display batch of images
    printBatch()