import torch 
import time 

import os
import sys 
sys.path.insert(0, os.getcwd()) 
#test import 
from src.model.UNet import UNet


class PytorchRuntime():
    def __init__(self,
        device,
        model,
        input_channel,
        input_height,
        input_width,
        iteration_number
    ) -> None :
        super(PytorchRuntime, self).__init__()
        self.device = device 
        self.model = model 
        self.input_channel = input_channel
        self.input_height = input_height
        self.input_width = input_width 
        self.iteration_number = iteration_number 

    
    def evaluate(self):
        input_list = []
        for i in range(self.iteration_number):
            input_list.append(torch.rand(1, self.input_channel, self.input_height, self.input_width, device=self.device))

        model = self.model.to(self.device)
        model.eval()
        start = time.time()
        for input in input_list:
            y = model(input)
        end = time.time()

        time_per_inf = round((end-start)/self.iteration_number, 3)
        fps = round(1/time_per_inf,3)
        print("Le temps d'execution du model est de {}s ({} FPS)sur {}".format(time_per_inf,fps,self.device))


def PytorchRuntime_test():
    DEVICE = torch.device("mps")
    INPUT_CHANNEL = 3 
    INPUT_HEIGHT = 572 
    INPUT_WIDTH = 572
    ITERATION_NUMBER = 10
    CLASS_NUMBER = 3

    model = UNet(INPUT_CHANNEL, CLASS_NUMBER)

    pytorchRuntime = PytorchRuntime(
        DEVICE,
        model,
        INPUT_CHANNEL,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        ITERATION_NUMBER
    )

    pytorchRuntime.evaluate()

if __name__ == "__main__":
    PytorchRuntime_test()