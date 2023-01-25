import torch
from torch.utils.data import Dataset
import h5py
from display import *
from helper import *

data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
# grandTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)
images = data.dataset['x']

image = data.dataset['x'][0]
display(image)
print("Has Cancer" if hasCancer(0) else "No Cancer")