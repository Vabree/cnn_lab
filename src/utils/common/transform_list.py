import torch 
import torchvision
from torchvision.transforms import ToTensor, Compose, Resize

def get_transform(transform_number:int):
    if transform_number == 0:
        transform = Compose([
                            ToTensor()  
        ])
        
    elif transform_number == 1:
        transform = Compose([
                        Resize((32,32)),
                        ToTensor()  
        ])

    elif transform_number == 2:
        transform = Compose([
            Resize((572,572)),
            ToTensor()
        ])

    elif transform_number == 3:
        transform = Compose([
            Resize((388,388)),
            ToTensor() 
        ])

    else:
        raise NameError("transform number doesn't exist")
        
    return transform
