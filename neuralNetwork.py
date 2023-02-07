import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import h5py
from torchvision.transforms import transforms
import random
from helper import plot
from display import *
from torch import Tensor




BATCH_SIZE = 100
TEST_SIZE = 32700
TRAIN_SIZE = 10000
number_of_labels = 2

# Create an instance for training. 
train_set_data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=BATCH_SIZE, shuffle=True)
train_set_labels = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=BATCH_SIZE, shuffle=True)

# Create an instance for testing, shuffle is set to False.
test_set_data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_x.h5', 'r'), batch_size=BATCH_SIZE, shuffle=False)
test_set_labels = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_y.h5', 'r'), batch_size=BATCH_SIZE, shuffle=False)

print("The number of images in the training set is: ", len(train_set_data.dataset['x']))
print("The number of images in the test set is: ", len(test_set_data.dataset['x']))

print("Training size is: ", TRAIN_SIZE)
print("Test size is: ", TEST_SIZE)
print("Batch size is: ", BATCH_SIZE)


classes = ('cancer', 'no cancer')

 # Define your execution device
device = torch.device("mps")
print("The model will be running on", device, "device")

# initialize the network weights with He
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

# This network applies two 2D convolutional layers to reduce the input size from 96x96 to 24x24, 
# and then applies max pooling to reduce it further to 8x8. The output is then vectorized and 
# passed through some fully connected layers before the final output is produced.

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #initialize the network weights
        self.apply(init_weights)
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
        self.fc4 = nn.Linear(128, 2).to(device)

    def forward(self, x):
        
        x = torch.permute(x, (0, 3, 1, 2)).to(device)
        # print("permute shape: ", x.shape)

        x = F.relu(self.bn1(self.conv1(x))).to(device)    
        # print("first conv shape: ", x.shape)
        
        x = F.relu(self.bn2(self.conv2(x))).to(device)
        # print("second conv shape: ", x.shape)   
        
        x = self.pool(x).to(device)
        # print("pool shape: ", x.shape)   
        
        x = F.relu(self.bn4(self.conv4(x))).to(device)
        # print("fourth conv shape: ", x.shape)
        
        x = self.pool(x).to(device)
        # print("pool shape: ", x.shape)

        x = F.relu(self.bn5(self.conv5(x))).to(device)   
        # print("fifth conv shape: ", x.shape)
        
        x = self.pool(x).to(device)
        # print("pool shape: ", x.shape)

        x = F.relu((self.conv6(x))).to(device)
        # print("sixth conv shape: ", x.shape)

        x = self.pool(x).to(device)

        x = x.view(x.size(0), -1).to(device)
        # print("view shape: ", x.shape)

        x = F.relu(self.fc1(x).to(device)).to(device)
        # print("fc1 shape: ", x.shape)

        x = F.relu(self.fc2(x).to(device)).to(device)
        # print("fc2 shape: ", x.shape)

        x = F.relu(self.fc3(x).to(device)).to(device)
        # print("fc3 shape: ", x.shape)


        x = self.fc4(x).to(device)
        # print("Model output values: ", x.shape)
        
        return x

model = Network()



loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)#, weight_decay=0.0001

def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

def printProgressBar (iteration, total, length, prefix = '', suffix = '', decimals = 1, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
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
        test_counter = 0
        inverted_counter = 0
        for j in range(0, TEST_SIZE, BATCH_SIZE):
            
            images = test_set_data.dataset['x'][j:j+BATCH_SIZE]
            labels = test_set_labels.dataset['y'][j:j+BATCH_SIZE]
            #add row to labels that inverts the labels
            labels = np.column_stack((labels, np.logical_not(labels)))
            # reshape the labels to be a 2x10 matrix
            labels = labels.reshape(BATCH_SIZE, 2)

            #convert images to tensor
            images = Variable(torch.tensor(images, dtype=torch.float32).to(device))
            #normalize the images
            images = images / 255.0
            labels = Variable(torch.tensor(labels, dtype=torch.float32).to(device))
            
            # run the model on the test set to predict labels
            outputs = model(images)
            # one-hot encode the outputs
            outputs = torch.nn.functional.one_hot(torch.argmax(outputs, dim=1), num_classes=2).to(device)
            
            #sum the elements of the first column
            predictedTrue = torch.sum(outputs[:,0])
            predictedFalse = torch.sum(outputs[:,1])
            



            # print(outputsMask)
            

            # print("test phase outputs: ", outputs)
            # print("test phase labels: ", labels)
            # Tensor.cpu(outputs)
            
            _, counts = torch.unique(outputs[outputs == labels], return_counts=True)
            # Tensor.cpu(counts)
            # print(outputs[outputs == labels])
            # print(counts)
            if len(counts) > 0:
                correctImages += counts[0].item()
            else:
                correctImages += 0


            # print("correct images count: ", correctImages)


            #compare the outputs to the labels
            outputs.unsqueeze(0) 
            # print((outputs == labels).all(dim=1).nonzero().size(0))

            # add to test_counter with batch size
            test_counter += BATCH_SIZE

            
                
            
                
            
                
            
            printProgressBar(test_counter, TEST_SIZE, prefix = ' Testing:', suffix = 'Complete', length = 10)

            
            
        #print prediction bias in the shape [---*---] where * is the bias
        print('Prediction bias: False [', end='')
        for i in range(int(predictedTrue/(predictedTrue+predictedFalse)*10)):
            print('-', end='')
        print('*', end='')
        for i in range(int(predictedFalse/(predictedTrue+predictedFalse)*10)):
            print('-', end='')
        print('] True')

        


        # compute the accuracy over all test images
        accuracy = correctImages / TEST_SIZE 
    print(correctImages, 'correct images out of', TEST_SIZE, 'images. Accuracy %.2f %%' % (correctImages/TEST_SIZE*100))
    # print('Epoch predicted True:', predictedTrue, ', False:', predictedFalse)
    # compute the accuracy over all test images
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
        print('Epoch ', epoch+1)
        running_loss = 0.0
        running_acc = 0.0
        batchCounter = 0
        
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
            images = Variable(torch.tensor(images, dtype=torch.float32).to(device))
            #normalize the images
            images = images / 255.0
            labels = Variable(torch.tensor(labels, dtype=torch.float32).to(device))
            # print("train phase labels print: ", labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images)
            # print("train phase outputs: ", torch.flatten(outputs))
            # print("train phase labels: ", torch.flatten(labels))

            # compute the loss
            loss = loss_fn(torch.flatten(outputs), torch.flatten(labels))
            loss.backward()
            optimizer.step()

            # Print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            # running_acc += (torch.flatten(outputs).round() == torch.flatten(labels)).sum().item()
            
            batchCounter = batchCounter + 1
            # print(" Epoch", epoch+1, '(', round( (batchCounter*BATCH_SIZE)/size*100), '%)', end='\r') 
            printProgressBar(batchCounter*BATCH_SIZE, TRAIN_SIZE, prefix = ' Training:', suffix = 'Complete', length = 50)           
            
            # if batchCounter/BATCH_SIZE % 1000 == 0:  
            #     # print every 1000  
            #     print('Epoch [%d/%d], Interval [%d/%d], Loss: %.4f'
            #         %(epoch+1, num_epochs, i+1, i+BATCH_SIZE, running_loss / 1000))

            #     # zero the loss
            #     running_loss = 0.0
            #     running_acc = 0.0
    
        accuracy = testAccuracy()
        # print('test accuracy over the test set is %f %%' % (accuracy*100))

        # Save the model if the accuracy is the best
        if accuracy > best_accuracy:
            print('Saving the model with accuracy %f %%' % (accuracy))
            saveModel()
            best_accuracy = accuracy
        accuracyPlot.append(accuracy)
        plot(accuracyPlot, None)
        #print loss
        print('Loss: %.4f' %( running_loss / TRAIN_SIZE))
        
        print()
        # print('training accuracy is %d %%' % (running_acc))
        
        


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


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test_set
    images = test_set_data.dataset['x'][0:BATCH_SIZE]
    labels = test_set_labels.dataset['y'][0:BATCH_SIZE]

    #convert entries to images

    images = Variable(torch.tensor(images, dtype=torch.float32))
    labels = Variable(torch.tensor(labels, dtype=torch.float32))

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
    # Build model
    train(200)
    print('Finished Training')

    # Test which classes performed well
    # testAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "myFirstModel.pth"

    model.load_state_dict(torch.load(path))
    print('Model loaded from %s' % path)
    # Test with batch of images
    testBatch()