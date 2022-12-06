import os
import sys 
sys.path.insert(0, os.path.join(os.getcwd(),"src/models"))

#from UNet import UNet
from LeNet5 import LeNet5
from UNet import UNet


def get_model(model_name:str, channel_number:int, class_number: int):
    if model_name == "LeNet5":
        model = LeNet5(class_number)
    elif model_name == "UNet":
        model = UNet(channel_number, class_number)
    else:
        raise NameError("Config model doesn't exist")

    return model