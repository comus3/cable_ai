import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid
from utils import *

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

def filter_model_output(output,score_threshold):
  filtred_output = list()
  for image in output:
    filtred_image = dict()
    for key in image.keys():
      filtred_image[key] = image[key][image['scores'] >= score_threshold]
    filtred_output.append(filtred_image)
  return filtred_output

def get_boolean_mask(output):
  for index,pred in enumerate(output):
    output[index]['masks'] = pred['masks'] > 0.5
    output[index]['masks'] = output[index]['masks'].squeeze(1)
  return output

def inspect_model_output(output):
  for index,prediction in enumerate(output):
    print(f'Input {index + 1} has { len(prediction.get("scores")) } detected instances')


trainingFolderPath = '/Users/lionelschinckus/Documents/cable_ai/docs/trainingData/tests'
fileNames = get_jpg_filenames(trainingFolderPath)
filePaths = []
for fileName in fileNames:
    filePaths.append(str(Path(trainingFolderPath) / fileName))

device = torch.device('cpu')
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights, progress=False).to(device)
model = model.eval()

image_list = []
for filePath in filePaths:
   image_list.append(resizeImageToSquare(filePath,320))

transforms = weights.transforms()
images = [transforms(d).to(device) for d in image_list]
for image in images:
  print(image.shape)

output = model(images) # list of dict
print(len(output)) # equals to how many images that were fed into the model
print(output[0].keys()) # dict_keys(['boxes', 'labels', 'scores', 'masks'])

score_threshold = .8
output = filter_model_output(output=output,score_threshold=score_threshold)
output = get_boolean_mask(output)
plotResizedImage([
    draw_segmentation_masks(image, prediction.get('masks'), alpha=0.9)
    for index, (image, prediction) in enumerate(zip(image_list, output))
])