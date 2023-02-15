import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import copy
import os
import torch
from PIL import Image
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import optim
from tqdm.notebook import trange, tqdm
import seaborn as sns; sns.set(style='whitegrid')
from api import logRun, logNetwork

# %matplotlib inline


labels_df = pd.read_csv('./histopathologic-cancer-detection/train_labels.csv')
# print(labels_df.head().to_markdown())

os.listdir('./histopathologic-cancer-detection/')

labels_df.shape

labels_df[labels_df.duplicated(keep=False)]

labels_df['label'].value_counts()

imgpath ="./histopathologic-cancer-detection/train/" # training data is stored in this folder
malignant = labels_df.loc[labels_df['label']==1]['id'].values    # get the ids of malignant cases
normal = labels_df.loc[labels_df['label']==0]['id'].values       # get the ids of the normal cases

# print('normal ids')
# print(normal[0:3],'\n')

# print('malignant ids')
# print(malignant[0:3])

def plot_fig(ids,title,nrows=5,ncols=15):

    fig,ax = plt.subplots(nrows,ncols,figsize=(18,6))
    plt.subplots_adjust(wspace=0, hspace=0) 
    for i,j in enumerate(ids[:nrows*ncols]):
        fname = os.path.join(imgpath ,j +'.tif')
        img = Image.open(fname)
        idcol = ImageDraw.Draw(img)
        idcol.rectangle(((0,0),(95,95)),outline='white')
        plt.subplot(nrows, ncols, i+1) 
        plt.imshow(np.array(img))
        plt.axis('off')

    plt.suptitle(title, y=0.94)
    plt.pause(1)


'''Print some examples of malignant and non-malignant cases'''
# plot_fig(malignant,'Malignant Cases')
# plot_fig(normal,'Non-Malignant Cases')

torch.manual_seed(0) # fix random seed

class pytorch_data(Dataset):
    
    def __init__(self,data_dir,transform,data_type="train"):      
    
        # Get Image File Names
        cdm_data=os.path.join(data_dir,data_type)  # directory of files
        
        file_names = os.listdir(cdm_data) # get list of images in that directory  
        idx_choose = np.random.choice(np.arange(len(file_names)), 
                                      4000,
                                      replace=False).tolist()
        # choose 4000 images in order
        # idx_choose = np.arange(4000)
        file_names_sample = [file_names[i] for i in idx_choose] # get the file names
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names_sample]   # get the full path to images
        
        # Get Labels
        labels_data=os.path.join(data_dir,"train_labels.csv") 
        labels_df=pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True) # set data frame index to id
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names_sample]  # obtained labels from df
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames) # size of dataset
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image) # Apply Specific Transformation to Image
        return image, self.labels[idx]

# transforms
data_transformer = transforms.Compose([transforms.ToTensor()])
                                      #  transforms.Resize((46,46))])

data_dir = './histopathologic-cancer-detection/'
img_dataset = pytorch_data(data_dir, data_transformer, "train") # Histopathalogic images                                       
print("dataset size:", len(img_dataset))
#load example tensor
img,label=img_dataset[10]
# print(img.shape,torch.min(img),torch.max(img))

len_img=len(img_dataset)
len_train=int(0.8*len_img)
len_val=len_img-len_train

# Split Pytorch tensor
train_ts,val_ts=random_split(img_dataset,
                             [len_train,len_val]) # random split 80/20

print("train dataset size:", len(train_ts))
print("validation dataset size:", len(val_ts))

# getting the torch tensor image & target variable
# ii=-1
# for x,y in train_ts:
#     print(x.shape,y)
#     ii+=1
#     if(ii>5):
#         break

# 
def plot_img(x,y,title=None):

    npimg = x.numpy() # convert tensor to numpy array
    npimg_tr=np.transpose(npimg, (1,2,0)) # Convert to H*W*C shape
    fig = px.imshow(npimg_tr)
    fig.update_layout(template='plotly_white')
    fig.update_layout(title=title,height=300,margin={'l':10,'r':20,'b':10})
    fig.show()

# Create grid of sample images
def plot_grid(grid_size, nrow, padding):

  rnd_inds=np.random.randint(0,len(train_ts),grid_size)
  print("image indices:",rnd_inds)

  x_grid_train=[train_ts[i][0] for i in rnd_inds]
  y_grid_train=[train_ts[i][1] for i in rnd_inds]

  x_grid_train=utils.make_grid(x_grid_train, nrow=nrow, padding=padding)
  print(x_grid_train.shape)
      
  plot_img(x_grid_train,y_grid_train,'Training Subset Examples')

  # Create grid of sample images
  rnd_inds=np.random.randint(0,len(val_ts),grid_size)
  print("image indices:",rnd_inds)
  x_grid_val=[val_ts[i][0] for i in range(grid_size)]
  y_grid_val=[val_ts[i][1] for i in range(grid_size)]

  x_grid_val=utils.make_grid(x_grid_val, nrow=nrow, padding=padding)
  print(x_grid_val.shape)

  plot_img(x_grid_val,y_grid_val,'Validation Dataset Preview')

# plot_grid(grid_size=30, nrow=10, padding=2)

# Define the following transformations for the training dataset
tr_transf = transforms.Compose([
#     transforms.Resize((40,40)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.RandomRotation(45),         
#     transforms.RandomResizedCrop(50,scale=(0.8,1.0),ratio=(1.0,1.0)),
    transforms.ToTensor()])

# For the validation dataset, we don't need any augmentation; simply convert images into tensors
val_transf = transforms.Compose([
    transforms.ToTensor()])

# After defining the transformations, overwrite the transform functions of train_ts, val_ts
train_ts.transform=tr_transf
val_ts.transform=val_transf

# The subset can also have transform attribute (if we asign)
train_ts.transform
batch_size = 64

# Training DataLoader
train_dl = DataLoader(train_ts,
                      batch_size=batch_size, 
                      shuffle=True)

# Validation DataLoader
val_dl = DataLoader(val_ts,
                    batch_size=batch_size,
                    shuffle=False)

# check samples
# for x,y in train_dl:
#     print(x.shape,y)
#     break


def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)



# Neural Network
class Network(nn.Module):
    
  # Network Initialisation
  def __init__(self, params):
      
      super(Network, self).__init__()
  
      Cin,Hin,Win=params["shape_in"]
      init_f=params["initial_filters"] 
      num_fc1=params["num_fc1"]  
      num_classes=params["num_classes"] 
      self.dropout_rate=params["dropout_rate"] 
      
      # Convolution Layers
      self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
      h,w=findConv2dOutShape(Hin,Win,self.conv1)
      self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
      h,w=findConv2dOutShape(h,w,self.conv2)
      self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
      h,w=findConv2dOutShape(h,w,self.conv3)
      self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
      h,w=findConv2dOutShape(h,w,self.conv4)
      self.conv5 = nn.Conv2d(8*init_f, 16*init_f, kernel_size=3)
      h,w=findConv2dOutShape(h,w,self.conv5)
      
      # compute the flatten size
      self.num_flatten=h*w*16*init_f
      self.fc1 = nn.Linear(self.num_flatten, num_fc1)
      self.fc2 = nn.Linear(num_fc1, 48)
      self.fc3 = nn.Linear(48, num_classes)

  def forward(self,X):
      
      # Convolution & Pool Layers
      X = F.leaky_relu(self.conv1(X),0.01).to(device)
      X = F.max_pool2d(X, 2, 2).to(device)
      X = F.leaky_relu(self.conv2(X),0.01).to(device)
      X = F.max_pool2d(X, 2, 2).to(device)
      X = F.leaky_relu(self.conv3(X),0.01).to(device)
      X = F.max_pool2d(X, 2, 2).to(device)
      X = F.leaky_relu(self.conv4(X),0.01).to(device)
      X = F.max_pool2d(X, 2, 2).to(device)
      X = F.leaky_relu(self.conv5(X),0.01).to(device)
      X = F.max_pool2d(X, 2, 2).to(device)

      X = X.view(-1, self.num_flatten).to(device)
      
      X = F.leaky_relu(self.fc1(X),0.01).to(device)
      X = F.dropout(X, self.dropout_rate).to(device)
      X = F.leaky_relu(self.fc2(X),0.01).to(device)
      X = F.dropout(X, self.dropout_rate).to(device)
      X = self.fc3(X).to(device)
      return F.log_softmax(X, dim=1).to(device)

# Neural Network Predefined Parameters
params_model={
        "shape_in": (3,96,96), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.5,
        "num_classes": 2}

# Create instantiation of Network class
cnn_model = Network(params_model)

# define computation hardware approach (GPU/CPU)
device = torch.device("mps") # if you have a GPU, change to "cuda"

model = cnn_model.to(device)

# summary(cnn_model, input_size=(3, 96, 96),device=device.type)

loss_func = nn.NLLLoss(reduction="sum")

opt = optim.Adam(cnn_model.parameters(), lr=0.0003)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)



''' Helper Functions'''

def printProgressBar (iteration, total, length, prefix = '', suffix = '', decimals = 1, fill = '█', printEnd = "\r"):
    
  percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
  filledLength = int(length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
  # Print new line on complete
  if iteration >= total: 
      print()

# L2 norm of the gradient of a batch of losses with respect to the parameters of the network
def gradient_norm(loss, model):
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** (1. / 2)
    return grad_norm

# Function to get the learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# Function to compute the loss value per batch of data
def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target) # get loss
    pred = output.argmax(dim=1, keepdim=True) # Get Output Class
    metric_b = pred.eq(target.view_as(pred)).sum().item() # get performance metric
    loss_each = loss.item()/len(target) # get loss per sample
    grad_norm = 0
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        # compute gradient norm
        grad_norm = gradient_norm(loss, model)
        
    return loss.item(), metric_b, grad_norm, loss_each

# Compute the loss value & performance metric for the entire dataset (epoch)
def loss_epoch(model,loss_func,dataset_dl,opt=None):
    
    run_loss=0.0 
    t_metric=0.0
    len_data=len(dataset_dl.dataset)
    counter=0
    batch_stats = []
    correlation = []
    # print(len_data, " samples in dataset")
    # internal loop over dataset
    for xb, yb in dataset_dl:
        
        # move batch to device
        # xb is batch of images ([64, 3, 96, 96])
        # yb is batch of labels ([64])
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb) # get model output ([64, 2])
        loss_b, metric_b, grad_n, loss_each = loss_batch(loss_func, output, yb, opt) # get loss per batch and gradient norm
        run_loss += loss_b        # update running loss
        
        #add grad_n and loss_each to batch_stats
        batch_stats.append([loss_each, grad_n])
        
        

        counter += 1
        printProgressBar(counter, len(dataset_dl), length=50)
        if metric_b is not None: # update running metric
            t_metric+=metric_b   
    
    # Correlation 
    correlation = np.corrcoef(batch_stats, rowvar=False)[0,1]
    # print(correlation)

    
        
    
    loss=run_loss/float(len_data)  # average loss value
    metric=t_metric/float(len_data) # average metric value
    
    return loss, metric, correlation

params_train={
  "train": train_dl,
  "val": val_dl,
  "epochs": 50,
  "optimiser": optim.SGD(cnn_model.parameters(), lr = 0.01, momentum=0.9),
  "lr_change": ReduceLROnPlateau(opt,
                                mode='min',
                                factor=0.5,
                                patience=20,
                                verbose=0),
  "f_loss": nn.NLLLoss(reduction="none"), #sum??
  "weight_path": "weights.pt",
  "check": False, 
}

def train_val(model, params):
  '''
  train_val returns:
  * The best performing model on the validation dataset
  * The Loss per iteration
  * The Evaluation Metric per iteration (accuracy)
  * The Correlation between the loss and the gradient norm
  '''
  # Get the parameters
  epochs=params["epochs"]
  loss_func=params["f_loss"]
  opt=params["optimiser"]
  train_dl=params["train"]
  val_dl=params["val"]
  lr_scheduler=params["lr_change"]
  weight_path=params["weight_path"]
  
  loss_history={"train": [],"val": []} # history of loss values in each epoch
  metric_history={"train": [],"val": []} # histroy of metric values in each epoch
  correlation_history={"train": [],"val": []} # histroy of correlation values in each epoch
  best_model_wts = copy.deepcopy(model.state_dict()) # a deep copy of weights for the best performing model
  best_loss=float('inf') # initialize best loss to a large value

  timestamps = []
  accuracyTrain = metric_history["train"]
  accuracyTest = metric_history["val"]
  lossTrain = loss_history["train"]
  lossTest = loss_history["val"]

  logNetwork(
      batch_size,
      len_val,
      'DCNN',
      get_lr(opt),
      'Adam',
      'NLLLoss',
      'foobar',
  )

  
  ''' Train Model n_epochs '''
  
  for epoch in tqdm(range(epochs), desc="Epochs"):

    ''' Get the Learning Rate '''
    current_lr=get_lr(opt)
    print('Epoch {}/{}, current lr={}'.format(epoch+1, epochs, current_lr), "\r")

    if epoch % 10 == 0:
        logRun(
            timestamps,
            accuracyTrain,
            accuracyTest,
            lossTrain,
            lossTest,
            'DCNN',
            5,
            'Random Sampling',
          )

    
    '''
    Train Model Process
    '''
    model.train()
    print("Training Model")
    train_loss, train_metric, train_correlation = loss_epoch(model,loss_func,train_dl,opt)

    loss_history["train"].append(train_loss)
    metric_history["train"].append(train_metric)
    correlation_history["train"].append(train_correlation)
    
    
    '''
    Evaluate Model Process
    '''
    model.eval()
    print("Evaluate Model")
    with torch.no_grad():
        val_loss, val_metric, val_correlation = loss_epoch(model,loss_func,val_dl)
    
    # store best model
    if(val_loss < best_loss):
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        
        # store weights into a local file
        torch.save(model.state_dict(), weight_path)
        print("Copied best model weights!")
    
    # collect loss and metric for validation dataset
    loss_history["val"].append(val_loss)
    metric_history["val"].append(val_metric)
    
    # learning rate schedule
    lr_scheduler.step(val_loss)
    if current_lr != get_lr(opt):
        print("Loading best model weights!")
        model.load_state_dict(best_model_wts) 

    
    print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}", "\r")
    # print("-"*10) 
    # printProgressBar(epoch, 50, prefix = 'Progress:', suffix = 'Complete', length = 50)
    print()
    # load best model weights
    model.load_state_dict(best_model_wts)

  return model, loss_history, metric_history, correlation_history


params_train={
  "train": train_dl,"val": val_dl,
  "epochs": 50,
  "optimiser": optim.Adam(cnn_model.parameters(),lr=0.0003),
  "lr_change": ReduceLROnPlateau(opt,
                                mode='min',
                                factor=0.5,
                                patience=20,
                                verbose=0),
  "f_loss": nn.NLLLoss(reduction="sum"),
  "weight_path": "weights.pt",
}

''' Actual Train / Evaluation of CNN Model '''

cnn_model, loss_hist, metric_hist, correlation_hist = train_val(cnn_model, params_train)
print("Training Complete!")

epochs=params_train["epochs"]

fig,ax = plt.subplots(1,3,figsize=(12,5), constrained_layout=True)
sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],ax=ax[0],label='Training Loss')
sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["val"],ax=ax[0],label='Validation Loss')
ax[0].set_title('Loss')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["train"],ax=ax[1],label='Training Accuracy')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["val"],ax=ax[1],label='Validation Accuracy')
ax[1].set_title('Accuracy')
sns.lineplot(x=[*range(1,epochs+1)],y=correlation_hist["train"],ax=ax[2],label='Training Correlation')
ax[2].set_title('Correlation')
plt.suptitle('Statistics of Training and Validation')
plt.pause(10)

