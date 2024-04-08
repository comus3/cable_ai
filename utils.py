import torch
from torchvision.transforms import v2
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from pathlib import Path


def get_jpg_filenames(folder_path):
    png_filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            png_filenames.append(filename)
    #print(png_filenames)
    return png_filenames

def resizeImageToSquare(inputImagePath, x):
    # Open the image using PIL
    image = Image.open(inputImagePath)
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((x, x)),  # Resize to x * x
    ])
    # Apply the transformation
    squareImage = transform(image)
    return squareImage

def plotResizedImage(resizedImage):
    # Convert PIL image to numpy array
    imgArray = resizedImage.convert('RGB')
    # Plot the image
    plt.imshow(imgArray)
    plt.axis('off')
    plt.show()

def testResize(filePaths):
    imgList = []
    for filePath in filePaths:
        resizedValues = resizeImageToSquare(filePath, 512)
        plotResizedImage(resizedValues)
        imgList.append(resizedValues)


def generate512(filePaths, thisPath):
    output_folder = 'outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    i = 0
    for filePath in filePaths:
        resizedValues = resizeImageToSquare(filePath, 512)  # Assuming resizeImageToSquare is a function that returns a PIL Image object
        resizedValues.save(os.path.join(output_folder, f"File{i}_IN_512.png")) 
        i += 1

def generate320(filePaths, thisPath):
    output_folder = 'outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    i = 0
    for filePath in filePaths:
        resizedValues = resizeImageToSquare(filePath, 320)  # Assuming resizeImageToSquare is a function that returns a PIL Image object
        resizedValues.save(os.path.join(output_folder, f"File{i}_IN_320.png")) 
        i += 1


