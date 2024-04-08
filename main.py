import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid
from utils import *




def init():
    global filePaths
    global thisPath
    thisPath = '/Users/lionelschinckus/Documents/cable_ai/git/cable_ai'
    trainingFolderPath = '/Users/lionelschinckus/Documents/cable_ai/docs/trainingData/masks'
    fileNames = get_jpg_filenames(trainingFolderPath)
    filePaths = []
    for fileName in fileNames:
        filePaths.append(str(Path(trainingFolderPath) / fileName))


    

init()
