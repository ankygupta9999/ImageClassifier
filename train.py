# Imports here
import argparse
import torch
from torchvision import datasets, transforms, models
import torchvision
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image
#import helper
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from collections import OrderedDict
import time
import random, os
import json
from utility import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default=3)
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train_model(model, epochs, gpu,criterion, optimizer, traindataloaders, validatedataloaders):
    device = torch.device("cuda" if gpu == "gpu" else "cpu")
    model.cuda()
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in traindataloaders:
            #move input and label tensors to available device(Cuda).
            images, labels = images.to(device), labels.to(device)
            #activate gradient calc
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            test_loss = 0
            accuracy = 0

            #turn off grad for validation
            with torch.no_grad():
                model.eval()
                for images, labels in validatedataloaders:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                model.train()
                train_losses.append(running_loss/len(traindataloaders))
                test_losses.append(test_loss/len(validatedataloaders))

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(traindataloaders)),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(validatedataloaders)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validatedataloaders)))

def train_save_model():
    
    args = parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    validate_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    traindataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testdataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    validatedataloaders = torch.utils.data.DataLoader(validate_datasets, batch_size=64, shuffle=True)
    
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    if args.arch == "densenet121":
        classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))
    elif args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(feature_num, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(1024, 102),
                                   nn.LogSoftmax(dim=1))
       
    criterion = nn.NLLLoss()
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))

    epochs = int(args.epochs)
    class_index = train_datasets.class_to_idx
    gpu = args.gpu
    
    train_model(model, epochs, gpu,criterion, optimizer, traindataloaders, validatedataloaders)
    
    model.class_to_idx = class_index
    path = args.save_dir # get the new save location 
    
    save_checkpoint(path, model, optimizer, args, classifier)

if __name__ == "__main__":
    train_save_model()