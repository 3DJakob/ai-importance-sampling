import torch
import h5py
grandTruth = torch.utils.data.DataLoader(h5py.File('./camelyonpatch_level_2_split_train_y.h5', 'r'), batch_size=32, shuffle=True)

def hasCancer(index):
    return grandTruth.dataset['y'][index][0][0][0]