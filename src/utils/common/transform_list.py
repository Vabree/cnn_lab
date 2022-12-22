import torch 
import torchvision
from torchvision.transforms import ToTensor, Compose, Resize, RandomCrop, PILToTensor, Normalize

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
    elif transform_number == 4:
        transform = Compose([
            Resize((256,256)),
            ToTensor() 
        ])
    elif transform_number == 5:
        transform = Compose([
            Resize((256,256)),
            PILToTensor()   
        ])
    elif transform_number == 6:

        transform = Compose([
            Resize((256,256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NameError("transform number doesn't exist")
        
    return transform
