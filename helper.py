import torch
import h5py
import matplotlib.pyplot as plt
from IPython import display


groundTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)

def hasCancer(index):
    return groundTruth.dataset['y'][index][0][0][0]


plt.ion()

def plot(meanCorrectlyClassified, correctlyClassified):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.ylim(ymin=0)
    if correctlyClassified is not None:
        plt.plot(correctlyClassified)
        plt.text(len(correctlyClassified)-1, correctlyClassified[-1], str(correctlyClassified[-1]))
    plt.plot(meanCorrectlyClassified)
    plt.text(len(meanCorrectlyClassified)-1, meanCorrectlyClassified[-1], str(meanCorrectlyClassified[-1]))
    plt.show(block=False)
    plt.pause(0.001)