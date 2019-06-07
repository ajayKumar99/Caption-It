import numpy as np


def Caption_Extract():
    train_captions = np.load('Model/checkpoints/train_captions.npy')
    return train_captions