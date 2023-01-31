import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import h5py
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 10
number_of_labels = 2

# Create an instance for training. 
train_set = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
train_set_y = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)



# Create a loader for the training set 
# shuffle true
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_set)*batch_size)

# Create an instance for testing, note that train is set to False.
test_set = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_x.h5', 'r'), batch_size=32, shuffle=False)
test_set_y = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_test_y.h5', 'r'), batch_size=32, shuffle=False)

# only use first 1000 images
# train_set = train_set.dataset['x'][0:10000]
# train_set_y = train_set_y.dataset['y'][0:10000]


# Create a loader for the test set 
# Note that each shuffle is set to false for the test loader.
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_set)*batch_size)

print("The number of batches per epoch is: ", len(train_set))
print(train_set.dataset['x'])
classes = ('cancer', 'no cancer')

 # Define your execution device
device = torch.device("mps")
print("The model will be running on", device, "device")

# This network applies two 2D convolutional layers to reduce the input size from 96x96 to 24x24, 
# and then applies max pooling to reduce it further to 8x8. The output is then vectorized and 
# passed through a fully connected layer before the final output is produced.

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(12).to(device)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(12).to(device)

        self.pool = nn.MaxPool2d(2, 2).to(device)
        
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1).to(device)
        self.bn4 = nn.BatchNorm2d(24).to(device)

        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1).to(device)
        self.bn5 = nn.BatchNorm2d(24).to(device)

        self.fc1 = nn.Linear(24*42*42, 1).to(device)

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2)).to(device)
        # print("permute shape: ", x.shape)

        x = F.relu(self.bn1(self.conv1(x))).to(device)    
        x = F.relu(self.bn2(self.conv2(x))).to(device)   
        x = self.pool(x).to(device)                        
        x = F.relu(self.bn4(self.conv4(x))).to(device)     
        x = F.relu(self.bn5(self.conv5(x))).to(device)   
          
        x = x.contiguous().view(x.size(0), -1).to(device)
        x = self.fc1(x).to(device)
        # print("Model output shape: ", x.shape)
        return x

model = Network()



loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        test_size = 2000
        for j in  range(0, test_size, batch_size):
            

            images = test_set.dataset['x'][j:j+batch_size]
            labels = test_set_y.dataset['y'][j:j+batch_size]


            #convert images to tensor
            images = Variable(torch.tensor(images, dtype=torch.float32).to(device))
            labels = Variable(torch.tensor(labels, dtype=torch.float32).to(device))
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

# Training function. 
# loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Running Training epoch", epoch+1, '...')
        running_loss = 0.0
        running_acc = 0.0
        # size of the first 10000 images
        size = 5000
        for i in range(0, size, batch_size):
            # get the inputs
            images = train_set.dataset['x'][i:i+batch_size]
            labels = train_set_y.dataset['y'][i:i+batch_size]
            
            # Convert to torch tensors with the float32 data type
            images = Variable(torch.tensor(images, dtype=torch.float32).to(device))
            labels = Variable(torch.tensor(labels, dtype=torch.float32).to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # print("images:___ ",images.shape)
            # forward + backward + optimize
            outputs = model(images)
            # print("flat: ", torch.flatten(outputs))
            loss = loss_fn(torch.flatten(outputs), torch.flatten(labels))#.unsqueeze(1).long()
            loss.backward()
            optimizer.step()
            
            # Print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            running_acc += (torch.flatten(outputs).round() == torch.flatten(labels)).sum().item()
            
            if i % 1000 == 0:  
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                print("running accuracy: ", running_acc/((i+1)*batch_size)*100, "%")
                # zero the loss
                running_loss = 0.0
                running_acc = 0.0
            
            
            

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        print("testing accuracy.. ")
        accuracy = testAccuracy()
        print()
        print('Epoch ', epoch+1)
        print('test accuracy over the test set is %d %%' % (accuracy/10))
        print('training accuracy is %d %%' % (running_acc))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy




# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_set))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))

if __name__ == "__main__":
    
    # Build model
    train(10)
    print('Finished Training')

    # Test which classes performed well
    testAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "model.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()