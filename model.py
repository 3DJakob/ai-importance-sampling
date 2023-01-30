import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torchshape import tensorshape

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")

class Linear_QNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    # conv2d the input image with size 96x96 with 3 channels (RGB)

    # outshape = tensorshape(nn.Conv2d(3, 32, kernel_size=2, stride=1).to(device), (1, 3, 96, 96))
    # print("outshape: ", outshape)
    # mer och mer kanaler lägre uplösning max pooling mellan
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1).to(device)

    # pooling layer 96x96 -> 48x48
    # pooling layer 46x46 -> 23x23
    self.pool = nn.MaxPool2d(2, 2).to(device)

    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1).to(device)
    self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1).to(device)
    self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1).to(device)

    # the output of the convolve is 23x23x64
    # Linear
    

    # self.linear1 = nn.Linear(23*23, 256).to(device)
    self.linear1 = nn.Linear(4*4 * 64, 64).to(device)
    self.linear2 = nn.Linear(64, output_size).to(device)
    print(output_size, "output size")
    


  def forward(self, x):
    # print("Got input of size: ", x.size())
    x = x.clone().detach().requires_grad_(True).to(device)
    # reshape from 96x96x3 to 3x96x96
    x = x.permute(2, 0, 1)
    
    # convolve the input image
    x = self.conv1(x)
    x = self.pool(x)

    # print("conv1: ", x.size())

    x = self.conv2(x)
    x = self.pool(x)

    x = self.conv3(x)
    x = self.pool(x)
    x = self.conv4(x)
    x = self.pool(x)
    # print("conv2: ", x.size())
    # flatten the output of the convolve
    # x = x.view(x.size(0), -1)

    # Add channels together?
    # x = torch.sum(x, -2)

    x = torch.flatten(x, 0)
    
    # print("Size inside: ", x.size())
    # print("conv1: ", x.size())
    # linear layer
    x = self.linear1(torch.squeeze(x, 0))
    # output layer
    x = self.linear2(torch.squeeze(x, 0))
    # print("Size out: ", x.size())
    return x


class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters(), lr=lr)
    self.criterion = nn.MSELoss()

  def train_step(self, state, action, reward, next_state, done):
    # state is size 96x96x3
    # action is size 1
    # reward is size 1
    
    # convert to tensors
    # state = torch.tensor(state, dtype=torch.float).to(device)
    # next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    state = state.clone().detach().requires_grad_(True).to(device)
    next_state = next_state.clone().detach().requires_grad_(True).to(device)
    action = torch.tensor(action, dtype=torch.float).to(device)
    reward = torch.tensor(reward, dtype=torch.float).to(device)



    # reshape state to 1x96x96x3
    # state = state.view(1, 3, 96, 96)
    # next_state = next_state.view(1, 3, 96, 96)

    # predict Q values for current state
    pred = self.model(state)
    # predict Q values for next state
    # next_pred = self.model.forward(next_state)

    # calculate target
    target = pred.clone().to(device)

    reward = torch.unsqueeze(reward, 0)
    # target should be torch.Size([1])
    
    target = target.clone().detach().requires_grad_(True).to(device)
    
    
    # 2: Q_new = r + y * max(next predicted Q values) - current predicted Q values

    self.optimizer.zero_grad()
    loss = self.criterion(target, pred)
    loss.backward()

    self.optimizer.step()

