import torch
import h5py
import matplotlib.pyplot as plt
from IPython import display
device = torch.device('cpu')

groundTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)

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

data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
images = data.dataset['x']
testSize = 1000

guesses = torch.zeros(testSize)


def getTestAnswers(size):
    answers = torch.zeros(size, 1).to(device)
    for i in range(0, size):
        answers[i] = hasCancer(i)
    return answers

def getTestData(size, randomize = True):       
    data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
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


def getTestDataVector(size, randomize = True):       
    data = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_x.h5', 'r'), batch_size=32, shuffle=True)
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
            dataIndex = torch.randint(1000, datasize, (1,)).item()
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
