import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torchshape import tensorshape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("mps")

class Linear_QNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    # conv2d the input image with size 96x96 with 3 channels (RGB)

    # outshape = tensorshape(nn.Conv2d(3, 32, kernel_size=48, stride=1), (1, 3, 96, 96))
    # print("outshape: ", outshape)
    self.conv1 = nn.Conv2d(3, 32, kernel_size=48, stride=1)
    # the output of the convolve is 23x23x32
    # Linear
    # 11 -> 23
    self.linear1 = nn.Linear(49*49, 256)
    self.linear2 = nn.Linear(256, output_size)
    


  def forward(self, x):
    # print("Got input of size: ", x.size())
    x = x.clone().detach().requires_grad_(True).to(device)
    # reshape from 96x96x3 to 3x96x96
    x = x.permute(2, 0, 1)
    
    # convolve the input image
    x = F.relu(self.conv1(x))
    # flatten the output of the convolve
    x = x.view(x.size(0), -1)
    # linear layer
    x = F.relu(self.linear1(x))
    # output layer
    x = self.linear2(x)
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

    # no long term memory
    
    # convert to tensors
    state = torch.tensor(state, dtype=torch.float).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    action = torch.tensor(action, dtype=torch.float).to(device)
    reward = torch.tensor(reward, dtype=torch.float).to(device)

    # reshape state to 1x96x96x3
    # state = state.view(1, 3, 96, 96)
    next_state = next_state.view(1, 3, 96, 96)

    # predict Q values for current state
    pred = self.model(state)
    # predict Q values for next state
    # next_pred = self.model(next_state)

    # calculate target
    target = pred.clone()

    # if game is over, target is just the reward
    if done:
      target[0][torch.argmax(action).item()] = reward
    else:
      # if game is not over, target is reward + gamma * max Q value of next state
      # target[0][torch.argmax(action).item()] = reward + self.gamma * torch.max(next_pred)
      target[0][torch.argmax(action).item()] = reward
