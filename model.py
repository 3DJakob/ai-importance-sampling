import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torchshape import tensorshape

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

class Linear_QNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    # conv2d the input image with size 96x96 with 3 channels (RGB)

    # outshape = tensorshape(nn.Conv2d(3, 32, kernel_size=2, stride=1).to(device), (1, 3, 96, 96))
    # print("outshape: ", outshape)
    # mer och mer kanaler lägre uplösning max pooling mellan
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1).to(device).requires_grad_(True)

    # pooling layer 96x96 -> 48x48
    # pooling layer 46x46 -> 23x23
    self.pool = nn.MaxPool2d(2, 2).to(device)

    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1).to(device).requires_grad_(True)
    self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1).to(device).requires_grad_(True)
    self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1).to(device).requires_grad_(True)

    # the output of the convolve is 23x23x64
    # Linear
    

    # self.linear1 = nn.Linear(23*23, 256).to(device)
    self.linear1 = nn.Linear(4*4 * 64, 64).to(device).requires_grad_(True)
    self.linear2 = nn.Linear(64, output_size).to(device).requires_grad_(True)

    # soft max layer
    self.softmax = nn.Softmax(dim=0).to(device).requires_grad_(True)
    


  def forward(self, x):
    # print("Got input of size: ", x.size())

    # x = x.clone().detach().requires_grad_(True).to(device)

    # reshape from 96x96x3 to 3x96x96
    x = x.permute(2, 0, 1)
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
    x = self.softmax(x)
    return x


class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters())
    # self.optimizer = optim.Adam(model.parameters(), lr=lr)

    self.criterion = nn.MSELoss()

  def train_step(self, state, action, hasCancer, next_state, done):
    # state is size 96x96x3
    # action is size 2
    # reward is size 1
    # print(action)

    # convert to tensors
    # state = state.clone().detach().requires_grad_(True).to(device)
    # next_state = next_state.clone().detach().requires_grad_(True).to(device)
    # action = action.clone().detach().requires_grad_(True).to(device)

    # Should we use action or get the action from the model?
    
    # pred = action
    pred = self.model(state)

    

    # 2: Q_new = r + y * max(next predicted Q values) - current predicted Q values
    # [0, 1] tensor
    target = torch.tensor([0, 0], dtype=torch.float).to(device).requires_grad_(True)
    if (hasCancer):
      target = torch.tensor([1, 0], dtype=torch.float).to(device).requires_grad_(True)
    else:
      target = torch.tensor([0, 1], dtype=torch.float).to(device).requires_grad_(True)


    # print(target, 'target')

    a = list(self.model.parameters())[0].clone()

    self.optimizer.zero_grad()
    loss = self.criterion(target, pred)
    # print(loss, 'loss')
    loss.backward()

    self.optimizer.step()

    b = list(self.model.parameters())[0].clone() 

    parameterDiff = torch.sum(torch.abs(a - b))
    print(parameterDiff, 'parameterDiff')

    # print(a[0][0][0])
    # print(b[0][0][0])
    # print('-----------------')

    # if a.flatten().tolist() == b.flatten().tolist():
    #   print('a==b')
    # else:
    #   print('a!=b')

    # print(list(self.model.parameters())[0].grad)
