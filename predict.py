# Imports here
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms, models

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict
import time
import random, os

from PIL import Image

import json
from utility import save_checkpoint, load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') # use a deafault filepath to a primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    
    img_pil = Image.open(image) # use Image
   
    adjust_image = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image = adjust_image(img_pil)
    
    return image

def predict(image_path, model, top_k=3, gpu='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #Load the Model
    #model = load_checkpoint('checkpoint1.pth')
    device = torch.device("cuda" if gpu == 'gpu' else "cpu")
    
    model_class = model.class_to_idx
    model.to(device)
    
    #img = Image.open(image_path)
    img = process_image(image_path)
    #img = torch.from_numpy(img)
    img = img.unsqueeze_(0)
    img = img.float()
    
    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img)
            
    #ps = F.softmax(output.data,dim=1) # use F
    ps = torch.exp(output)

    probability = np.array(ps.topk(int(top_k))[0][0])
    
    index_to_classes = {val: key for key, val in model.class_to_idx.items()}
    top_classes =  [np.int(index_to_classes[each])  for each in np.array(ps.topk(int(top_k))[1][0])]
     
    return probability, top_classes   

def PredictTheFlower():
    #Load the args
    args = parse_args()
    #Load the Model
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    path = args.filepath
    probs, classes = predict(path, model, args.top_k, args.gpu)
    flowername =  [cat_to_name[str(index)] for index in classes ]
    probabilities = probs
    
    print('selects file : ' + path) 
    print('flower names : ', flowername)
    print('probabilities : ', probabilities)
    
    # prints out classes corrspoding to probs 
    j=0 
    while j < len(flowername):
        print("{} has the probability of {}".format(flowername[j], probabilities[j] ))
        j = j+1
    
if __name__ == "__main__":
     PredictTheFlower()