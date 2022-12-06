import os 

import torch
from torch.utils.data import Dataset

import torchvision 
import torchvision.transforms.functional as T

from PIL import Image

class VocSegmentation(Dataset):
    def __init__(self,
        root: str,
        transform: T = None
    ) -> None:
        self.root = root
        self.transform = transform
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.mask = list(sorted(os.listdir(os.path.join(root, "SegmentationClass"))))
        print(len(self.imgs))
        print(len(self.mask))

        assert (len(self.imgs) == len(self.mask))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "SegmentationClass", self.mask[idx])
        
        img = Image.open(img.path).convert('RGB')
        target = Image.open(mask_path)
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target


def VocSegmentation_test():
    dataset = VocSegmentation("data/VOCdevkit/VOC2012")
if __name__ == "__main__":
    VocSegmentation_test()