#import functions
from Train_args import get_input_args
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import json
with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#predict 
in_arg = get_input_args()
def loadcheckpoint(filepath) :
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for param in model.parameters():
        param.requires_grad = False
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers_input'])),
                          ('relu', nn.ReLU()),
                          ('droupout1', nn.Dropout(checkpoint['drop'])),
                          ('fc2', nn.Linear(checkpoint['hidden_layers_input'], checkpoint['hidden_layers_output'])),
                          ('relu', nn.ReLU()),
                          ('droupout2', nn.Dropout(checkpoint['drop'])),
                          ('fc3', nn.Linear(checkpoint['hidden_layers_output'], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.classifier.optimizer = checkpoint['optimizer']
    model.classifier.epochs = checkpoint['epochs']
    model.classifier.learning_rate = checkpoint['learning_rate']
    model.classifier.class_to_idx = checkpoint['class_to_idx']

    
    return model

model = loadcheckpoint(in_arg.load_dir)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # resize and crop image

    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)

def predict(image_path, model, topk=in_arg.nb_topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.cuda()
    model.eval()

    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float().cuda()
    output = model(image)
    
    probabilities = torch.exp(output).data
    
    probs = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    

    index_to_class = {val: key for key, val in model.classifier.class_to_idx.items()} #get class names from dict
    top_classes = [index_to_class[each] for each in index]
    
    class_names = []
    for x in top_classes :
        class_names.append(cat_to_name[x])

    return probs, top_classes, class_names

probs, top_classes, class_names = predict(in_arg.dir_image, model, topk=in_arg.nb_topk)
print(probs)
print(top_classes)
print(class_names)


