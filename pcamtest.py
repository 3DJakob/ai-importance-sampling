import torchvision
import torch
from torch.utils.data import Dataset, DataLoader

# Define the transform to preprocess the data (you can modify it according to your needs)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convert PIL image to Tensor
    torchvision.transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

# Load the PCAM dataset
dataset = torchvision.datasets.PCAM(root='./data', download=True, transform=transform)

# Create a DataLoader to iterate over the dataset
batch_size = 64  # Modify this according to your needs



class MyDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
  def __getitem__(self, index):
    # data = self.dataset[index] 
    # target = self.targets[index]
    data, target = self.dataset[index] 
    return data, target, index

  def __len__(self):
    return len(self.dataset)
  


custom_dataset = MyDataset(dataset)
train_loader = DataLoader(custom_dataset,
                    batch_size=batch_size, shuffle=True)


# Iterate over every sample in the train_loader
for batch_idx, (data, target, idx) in enumerate(train_loader):
    # data contains the input images
    # target contains the corresponding labels
    # idx contains the indices of the samples in the current batch

    print(idx, 'idx')

    if batch_idx > 0:
        break

