import torch
import torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module):

    def __init__(self,in_channels=3,num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Layer 2
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Layer 3
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        #Spatial Adaptive Pooling to handle variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )


    def forward(self,x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
    

class ResnetTransfer(nn.Module):

    def __init__(self,num_classes=10, freeze_backbone=True):
        super().__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")

        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # remove avgpool and fc layers

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        #CNN Head

        self.cnn_head = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )


    def forward(self,x):
        x = self.backbone(x)
        x = self.cnn_head(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x