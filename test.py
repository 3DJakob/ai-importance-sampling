import torch
import torch.nn as nn
import torch.optim as optim
from helper import getTestAnswers, getTestData, plot

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
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # the output of the convolve is 23x23x64
        # Linear
        self.linear1 = nn.Linear(4*4 * 64, 64)
        self.linear2 = nn.Linear(64, output_size)
        # soft max layer
        # self.softmax = nn.Softmax(dim=0)

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


BATCH_SIZE = 64

# [data, target] = getTestData(64, True)
# model = StatusNN(3 * 96 * 96, 512) # 27648
model = Linear_QNet(3 * 96 * 96, 512) # 27648
# target = getTestAnswers(64)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

accuracyPlot = []
lastMeanAccuracy = 0
meanAccuracyPlot = []

for epoch in range(10000):
    
    [data, target] = getTestData(BATCH_SIZE, True)
    for i in range(BATCH_SIZE):
        optimizer.zero_grad()
        # print("Epoch: ", epoch)

        # forward pass
        # print("Input size: ", data[i].size())
        output = model(data[i])
        # print("Output size: ", output.size())
        # if target is one [0, 1] if target is zero [1, 0]
        t = torch.zeros(2)
        t[int(target[i])] = 1

        loss = criterion(output, t)
        # print("Loss: ", loss.item())
        
        # backward pass
        loss.backward()
        optimizer.step()

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


    accuracy = 0
    # test model
    with torch.no_grad():
        accuracy= 0
        for i in range(BATCH_SIZE):
            [data, target] = getTestData(64, False)
            output = model(data[i])
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            accuracy = accuracy + (output == target[i]).float().mean()
        accuracy = accuracy / BATCH_SIZE
        # print("Accuracy: ", accuracy)
        accuracyPlot.append(accuracy)
        # tenLastAccuracy = accuracyPlot[-10:]
        meanAccuracyPlot.append(sum(accuracyPlot[-100:]) / len(accuracyPlot[-100:]))
        plot(accuracyPlot, meanAccuracyPlot)
        
        # [data, target] = getTestData(64, False)
        # output = model(data[i])
        # output = torch.sigmoid(output)
        # output = (output > 0.5).float()
        # accuracy = (output == target[i]).float().mean()
        # print("Accuracy: ", accuracy)
        # accuracyPlot.append(accuracy)
        # plot(accuracyPlot, None)
    


    # accuracy = 0
    # # test model
    # with torch.no_grad():
    #     output = model(data)
    #     output = torch.sigmoid(output)
    #     output = (output > 0.5).float()
    #     accuracy = (output == target).float().mean()
    #     print("Accuracy: ", accuracy)

