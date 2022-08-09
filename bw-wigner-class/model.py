# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:09:41 2022

@author: tnsak
"""

'''
QUESTIONS:
    How do I say that X-axis does not matter but Y-axis does
    Is it possible to do an event style model? This type of data (EC trains)
    is unique in that we are confident each event is 1 species, and they
    are all related. We really only care about classification at the event
    level, not at the detection level. 
TODO:
    Change class layer to 5 classes
'''

import torch.nn as nn
from torchvision.models import resnet50

class BeakerNet(nn.Module):

    def __init__(self, num_classes):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(BeakerNet, self).__init__()

        self.feature_extractor = resnet50(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one
    

    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-18 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction