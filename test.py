import torch
import torch.nn as nn
import torch.optim as optim
from helper import getTestAnswers, getTestData

class StatusNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)      # [batch_size, 512] 
        x = self.relu(x)     # [batch_size, 512]
        x = self.fc2(x)      # [batch_size, 64]
        x = self.relu(x)     # [batch_size, 64]
        x = self.fc3(x)      # [batch_size, 1]
        return x

BATCH_SIZE = 1

[data, target] = getTestData(64, BATCH_SIZE)
model = StatusNN(BATCH_SIZE * 3 * 96 * 96, 512) # 27648
# target = getTestAnswers(64)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    optimizer.zero_grad()
    print("Epoch: ", epoch)

    # forward pass
    print("Input size: ", data.size())
    output = model(data)
    print("Output size: ", output.size())
    loss = criterion(output, target)
    print("Loss: ", loss.item())
    
    # backward pass
    loss.backward()
    optimizer.step()


    accuracy = 0
    # test model
    with torch.no_grad():
        output = model(data)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        accuracy = (output == target).float().mean()
        print("Accuracy: ", accuracy)

