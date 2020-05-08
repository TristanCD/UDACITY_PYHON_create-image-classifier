#import functions

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
from PIL import Image

from Train_args import get_input_args

def main():
    in_arg = get_input_args()

    #Load and process the data
    data_dir = in_arg.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    #Load the datasets with ImageFolder
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    #label mapping
    import json

    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    #load model
    if in_arg.arch == 'vgg16': 
        arch = 'vgg16'
        model = models.vgg16(pretrained=True)
        input_units = 25088
        model
    elif in_arg.arch == 'densenet121':
        arch = 'densenet121'
        model = models.densenet121(pretrained=True)
        input_units = 1024
        model
   


    #building and training the network

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    for param in model.parameters():
        param.requires_grad = False
        
    if in_arg.gpu == True and torch.cuda.is_available():
        print('On GPU \n')
    else:
        print('On CPU \n')
        
    #load arguments
    hidden_units = in_arg.hidden_units
    drop_rate = in_arg.drop_rate
    learning_rate = in_arg.learning_rate

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('droupout1', nn.Dropout(drop_rate)),
                              ('fc2', nn.Linear(hidden_units, 1048)),
                              ('relu', nn.ReLU()),
                              ('droupout2', nn.Dropout(drop_rate)),
                              ('fc3', nn.Linear(1048, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 60
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                        f"Valid accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
    #testing the network
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    running_loss = 0

    # Save the checkpoint 
    model.classifier.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': input_units,
                  'output_size': 102,
                  'hidden_layers_input': hidden_units,
                  'hidden_layers_output': 1048,
                  'drop': drop_rate,
                  'learning_rate': learning_rate,
                  'arch': arch,
                  'classifier' : classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.classifier.class_to_idx,
                  'state_dict': model.classifier.state_dict()}

    torch.save(checkpoint, in_arg.save_dir)

if __name__ == "__main__":
    main()

