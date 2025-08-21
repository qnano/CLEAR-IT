# clearit/models/mobilenet.py
from .encoder import BaseEncoder
#from torchvision.models import mobilenet_v2
#import torch.nn as nn

class MobileNetEncoder(BaseEncoder): # not implemented yet
    def __init__(self, output_size):
        super().__init__()
        # Initialize MobileNet similarly, adjusting layers and MLP as needed

    def forward(self, x):
        # Implement forward pass specific to MobileNet
        pass