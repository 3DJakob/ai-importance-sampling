import torch
import torch.nn as nn
import torch.optim as optim
from helper import getTestAnswers, getTestData, plot, getTestDataVector
import torchvision
import torchvision.transforms as transforms
from torchvision.models.vgg import VGG16_Weights

device = torch.device("cpu")

class StatusNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("x size: ", x.size())
        x = x.flatten(start_dim=1)
        x = self.fc1(x)      # [batch_size, 512] 
        x = self.relu(x)     # [batch_size, 512]
        x = self.fc2(x)      # [batch_size, 64]
        x = self.relu(x)     # [batch_size, 64]
        x = self.fc3(x)      # [batch_size, 1]
        return x

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super().__init__()
        output_size = 2
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, device=device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, device=device)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, device=device)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, device=device)
        # the output of the convolve is 23x23x64
        # Linear
        self.linear1 = nn.Linear(4*4 * 64, 64, device=device)
        self.linear2 = nn.Linear(64, output_size, device=device)
        # soft max layer
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = torch.flatten(x, 0)
        x = self.linear1(torch.squeeze(x, 0))
        x = self.linear2(torch.squeeze(x, 0))
        # x = self.softmax(x)
        return x

class VGG(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super().__init__()
        output_size = 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, device=device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, device=device)
        self.linear1 = nn.Linear(15488, 128, device=device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, output_size, device=device)
        # self.softmax = nn.Softmax(dim=0)
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 0)
        x = self.linear1(torch.squeeze(x, 0))
        x = self.relu(x)
        x = self.linear2(torch.squeeze(x, 0))
        x = self.softmax(x)
        return x


BATCH_SIZE = 1000
TESTSIZE = 1000

# [data, target] = getTestData(64, True)
# model = StatusNN(3 * 96 * 96, 512) # 27648
# model = VGG16(3 * 96 * 96, 512) # 27648
# model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model = Linear_QNet(3 * 96 * 96, 512) # 27648
# target = getTestAnswers(64)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

accuracyPlot = []
lastMeanAccuracy = 0
meanAccuracyPlot = []
predictionRatioPlot = []

for epoch in range(10000):
    model.train()
    
    [data, target] = getTestDataVector(BATCH_SIZE, True)
    meanLoss = 0
    for i in range(BATCH_SIZE):
        optimizer.zero_grad()
        # print("Epoch: ", epoch)
        # img = transforms.ToPILImage()(data[i])
        # img = transforms.ToTensor()(img)
        output = model(data[i])
        loss = criterion(output, target[i])
        
        # print("Loss: ", loss.item())
        meanLoss += loss.item()
        
        # backward pass
        loss.backward()
        optimizer.step()
    meanLoss /= BATCH_SIZE
    print("Mean Loss: ", meanLoss)
    # [data, target] = getTestData(64, True)
    # optimizer.zero_grad()
    # print("Epoch: ", epoch)

    # # forward pass
    # print("Input size: ", data.size())
    # output = model(data)
    # print("Output size: ", output.size())
    # loss = criterion(output, target)
    # print("Loss: ", loss.item())
    
    # # backward pass
    # loss.backward()
    # optimizer.step()

    model.eval()
    accuracy = 0
    predictionRatio = 0
    # test model
    with torch.no_grad():
        print("Epoch: ", epoch)
        accuracy= 0
        [data, target] = getTestDataVector(TESTSIZE, False)
        for i in range(TESTSIZE):
            output = model(data[i])
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            if output[0] == 1:
                predictionRatio = predictionRatio + 1
            accuracy = accuracy + (output == target[i]).float().mean()
        accuracy = accuracy / TESTSIZE
        accuracy = accuracy.item()
        # print("Accuracy: ", accuracy)
        accuracyPlot.append(accuracy)
        # tenLastAccuracy = accuracyPlot[-10:]
        # meanAccuracyPlot.append(sum(accuracyPlot[-100:]) / len(accuracyPlot[-100:]))
        predictionRatio = predictionRatio / TESTSIZE
        predictionRatioPlot.append(predictionRatio)
        plot(accuracyPlot, predictionRatioPlot)
    

