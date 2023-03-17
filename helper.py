import torch
import h5py
import matplotlib.pyplot as plt
from IPython import display
import torchvision.datasets as dsets

device = torch.device('cpu')

groundTruth = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)

def hasCancer(index):
    return groundTruth.dataset['y'][index][0][0][0]


plt.ion()

def plot(meanCorrectlyClassified, correctlyClassified):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Batches')
    plt.ylabel('Score')
    plt.ylim(ymin=0)
    if correctlyClassified is not None:
        plt.plot(correctlyClassified)
        plt.text(len(correctlyClassified)-1, correctlyClassified[-1], str(correctlyClassified[-1]))
    plt.plot(meanCorrectlyClassified)
    plt.text(len(meanCorrectlyClassified)-1, meanCorrectlyClassified[-1], str(meanCorrectlyClassified[-1]))
    plt.show(block=False)
    plt.pause(0.001)

def evaluate(agent):
    identical = True
    # no grad
    with torch.no_grad():

        correctlyClassified = 0
        for i in range(0, 1000):
            image = images[i]
            # action = agent.getAction(image)

            state0 = torch.tensor(image, dtype=torch.float)
            prediction = agent.model(state0)
            # print(prediction, 'prediction')
            # action = prediction.item()
            action = prediction
            # final_move = move

            if action[0] > 0.5 and hasCancer(i):
                correctlyClassified += 1
            elif action[0] <= 0.5 and not hasCancer(i):
                correctlyClassified += 1

            
            if (guesses[i] != action[0]):
                identical = False
            guesses[i] = action[0]
            
            # if i % 100 == 0:
            #     print(action, 'action')


        print('Idential guesses: ', identical)
        return correctlyClassified / 1000

data = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
images = data.dataset['x']
testSize = 1000

guesses = torch.zeros(testSize)


def getTestAnswers(size):
    answers = torch.zeros(size, 1).to(device)
    for i in range(0, size):
        answers[i] = hasCancer(i)
    return answers

def getTestData(size, randomize = True):       
    data = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
    # groundTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)
    images = data.dataset['x']

    # Flat all images
    data = torch.zeros(size, 3, 96, 96).to(device) # 64 samples, 3 channels, 96x96 image
    target = torch.zeros(size, 1).to(device) # 64 samples, 10 correct answers

    for i in range(0, size):
        dataIndex = i
        # print('randomize', randomize)
        if randomize == True:
            datasize = images.shape[0]
            dataIndex = torch.randint(1000, datasize, (1,)).item()
        # print(dataIndex, 'dataIndex')
        img = images[dataIndex]
        tensorData = torch.tensor(img, dtype=torch.float)
        # permute [96, 96, 3] to [3, 96, 96]
        tensorData = tensorData.permute(2, 0, 1)
        # insert into data
        data[i] = tensorData
        target[i] = hasCancer(dataIndex)

def getTestMnistData(size):
    # load MNIST dataset
    trainset = dsets.MNIST(root='./data', train=True, download=True, transform=None)
    images = trainset.data
    targets = trainset.targets

    # convert targets to one-hot
    labels = torch.zeros(targets.shape[0], 10)
    labels[torch.arange(targets.shape[0]), targets] = 1
    targets = labels

    # pick size random images
    data = torch.zeros(size, 1, 28, 28).to(device) # 64 samples, 28x28 image
    target = torch.zeros(size, 10).to(device) # 64 samples, 10 different classes

    for i in range(0, size):
        dataIndex = torch.randint(0, images.shape[0], (1,)).item()
        data[i] = images[dataIndex]
        target[i] = targets[dataIndex]

    target = torch.nn.functional.one_hot(torch.argmax(target, dim=1), num_classes=10).float()
    return [data, target]

def getEvaluationMnistData(size):
    # load MNIST test dataset
    testset = dsets.MNIST(root='./data', train=False, download=True, transform=None)
    images = testset.data
    targets = testset.targets

    # convert targets to one-hot
    labels = torch.zeros(targets.shape[0], 10)
    labels[torch.arange(targets.shape[0]), targets] = 1
    targets = labels
    targets = targets[0:size]
    # to tensor
    targets = torch.tensor(targets, dtype=torch.float)


    data = images[0:size]
    # transorm to sizex28x28 to sizex1x28x28
    data = data.unsqueeze(1)
    data = torch.tensor(data, dtype=torch.float)
    # pick first size images
    # target = torch.nn.functional.one_hot(torch.argmax(target, dim=1), num_classes=10).float()
    return [data, targets]


def getTestDataVector(size, randomize = True):       
    data = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
    # groundTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)
    images = data.dataset['x']

    # Flat all images
    data = torch.zeros(size, 3, 96, 96).to(device) # 64 samples, 3 channels, 96x96 image
    target = torch.zeros(size, 2).to(device) # 64 samples, 10 correct answers

    for i in range(0, size):
        dataIndex = i
        # print('randomize', randomize)
        if randomize == True:
            datasize = images.shape[0]
            dataIndex = torch.randint(0, datasize, (1,)).item()
        # print(dataIndex, 'dataIndex')
        img = images[dataIndex]
        tensorData = torch.tensor(img, dtype=torch.float)
        # permute [96, 96, 3] to [3, 96, 96]
        tensorData = tensorData.permute(2, 0, 1)
        # Scale to 0-1
        tensorData = tensorData / 255
        # insert into data
        data[i] = tensorData
        if hasCancer(dataIndex) == 1:
            target[i] = torch.tensor([0, 1])
        else:
            target[i] = torch.tensor([1, 0])    
    
    
    return [data, target]


def getEvaluationData(size):
    data = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_test_x.h5', 'r'), batch_size=32, shuffle=True)
    groundTruth = torch.utils.data.DataLoader(h5py.File('./data/camelyonpatch_level_2_split_test_y.h5', 'r'), batch_size=32, shuffle=True)
    
    images = data.dataset['x']
    
    if size == None:
        size = images.shape[0]

    # Flat all images
    data = torch.zeros(size, 3, 96, 96).to(device)
    target = torch.zeros(size, 2).to(device)

    for i in range(0, size):
        img = images[i]
        tensorData = torch.tensor(img, dtype=torch.float)
        tensorData = tensorData.permute(2, 0, 1)

        data[i] = tensorData
        if (groundTruth.dataset['y'][i][0][0][0] == 1):
            target[i] = torch.tensor([0, 1])
        else:
            target[i] = torch.tensor([1, 0])
    
    return [data, target]