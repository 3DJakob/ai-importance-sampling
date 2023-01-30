import torch
from torch.utils.data import Dataset
import h5py
from display import *
from helper import hasCancer, plot
from model import Linear_QNet, QTrainer
from collections import deque
import random

data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
# grandTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)
images = data.dataset['x']

image = data.dataset['x'][0]
display(image)
print("Has Cancer" if hasCancer(0) else "No Cancer")

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
IMG_SIZE = 96
HIDDEN_SIZE = 96 * 96

class Agent:
  def __init__(self):
    self.n_iterations = 0
    self.epsilon = 0 # randomness
    self.gamma = 0.99 # discount rate
    self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
    self.model = Linear_QNet(IMG_SIZE * IMG_SIZE * 3, HIDDEN_SIZE, 1)

    self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    self.correctlyClassified = 0

  def imageToState(self, image):
    return image.flatten()

  def getState(self, image):
    return torch.tensor(self.imageToState(image), dtype=torch.float)

  def trainShortMemory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)

  def getAction(self, state):
    # random moves: tradeoff exploration / exploitation
    self.epsilon = 80 - self.n_iterations
    final_move = 0
    if random.randint(0, 200) < self.epsilon:
      move = random.randint(0, 100)
      final_move = move / 100
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      final_move = move

    return final_move

  def train_short_memory(self, state, action, reward, next_state, game_over):
    self.trainer.train_step(state, action, reward, next_state, game_over)

def train():
  agent = Agent()

  classifyPlot = []
  rewardLast100 = []
  meanLast100 = []
  rewardPlot = []

  while True:
    randomIndex = random.randint(0, len(images) - 1)
    # image = random.choice(images)
    image = images[randomIndex]
    flat = agent.imageToState(image)
    # final_move = agent.getAction(flat)
    final_move = agent.getAction(image)
    move = 0
    reward = 0
    if final_move > 0.5:
      move = 1
    else :
      move = 0
    if (move == hasCancer(randomIndex)):
      reward = 1
    else:
      reward = 0
    # print("Move: " + str(move) + " Reward: " + str(reward) + " Has Cancer: " + str(hasCancer(randomIndex)))
    
    # Train short memory
    state_new = agent.getState(image) # has no effect
    # state_old = agent.getState(image)
    state_old = torch.tensor(image, dtype=torch.float)
    agent.train_short_memory(state_old, final_move, reward, state_new, False)

    agent.n_iterations += 1
    if agent.n_iterations % 100 == 0:
      print("Iteration: " + str(agent.n_iterations) + " Reward: " + str(reward) + " Has Cancer: " + str(hasCancer(randomIndex)))
    
    # Uncomment to see the plot
    # agent.correctlyClassified += reward
    # classifyPlot.append(agent.correctlyClassified / agent.n_iterations)
    # rewardLast100.append(reward)
    # # if larger that 100 then remove the first element
    # if len(rewardLast100) > 100:
    #   rewardLast100.pop(0)
    # rewardPlot.append(reward)
    # meanLast100.append(sum(rewardLast100) / len(rewardLast100))
    # plot(classifyPlot, meanLast100)
    


if __name__ == '__main__':
  train()
  # image = random.choice(images)
  # agent = Agent()
  # action = agent.getAction(image)
  # print(action)