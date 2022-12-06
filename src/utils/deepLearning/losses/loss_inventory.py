import os
import sys 
import argparse
sys.path.insert(0, os.path.join(os.getcwd()))

import torch 
import torch.nn as nn

def get_loss(loss_name:str):
    if loss_name == "CrossEntropyLoss":
        loss = nn.CrossEntropyLoss()
    else:
        raise NameError("loss doesn't exist")
    
    return loss

