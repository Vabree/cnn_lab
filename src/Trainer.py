import os
import sys 
import argparse
sys.path.insert(0, os.path.join(os.getcwd()))

import torch 
import torch.nn as nn

import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

import torchvision
from torchvision.transforms import ToTensor, Compose, Resize

import plotly.express as px
import pandas as pd

from tqdm.auto import tqdm

from src.models.model_inventory import get_model
from src.utils.deepLearning.metrics.metric_inventory import get_metric


class Trainer:
    def __init__(self,
            epochs: int,
            model: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloarder: torch.utils.data.DataLoader,
            criterion: torch.nn.Module,
            optimizer: torch.nn.Module,
            metrics: torchmetrics.Metric,
            device
        ) -> None:

        super(Trainer,self).__init__()
        self.device = device

        self.epochs = epochs 
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloarder
        self.criterion = criterion 
        self.optimizer = optimizer

        self.train_metrics = metrics.clone(prefix='train_').to(self.device)
        self.test_metrics = metrics.clone(prefix='test_').to(self.device)

        self.training_loss = []
        self.testing_loss = [] 

    def training_loop(self)->None:
        self.model.train()

        running_loss = 0
        with tqdm(self.train_dataloader, unit="batch") as tepoch:
            for images, labels in tepoch:  
                images = images.to(self.device)
                labels = labels.to(self.device)

                #Get ride of dimensions of size 1 to match the loss size wanted
                labels = labels.squeeze(dim=1) 

                self.optimizer.zero_grad()

                #Forward pass
                preds = self.model(images)
                loss = self.criterion(preds, labels.long())

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                self.train_metrics(preds, labels)
                running_loss += loss.item() * images.size(0)
                tepoch.set_postfix(loss = loss.item())
                
        self.training_loss.append(running_loss / len(self.train_dataloader.dataset))

    def testing_loop(self)->None:
        self.model.eval()

        running_loss = 0
        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                labels = labels.squeeze(dim=1) 
                preds = self.model(images)
                loss = self.criterion(preds, labels.long())
                self.test_metrics(preds,labels)

                running_loss += loss.item() * images.size(0)
        self.testing_loss.append(running_loss / len(self.test_dataloader.dataset))

    def run(self, save_name:str = None)->None:
        for epoch in range(self.epochs):
            print("--- Epoch {}".format(epoch+1))

            self.training_loop()
            train_acc = self.train_metrics.compute()
            print("- Metrics on train dataset : {}".format(train_acc))

            print("Testing ...",end="\r")
            self.testing_loop()
            test_acc = self.test_metrics.compute()
            print("- Metrics on test dataset : {}".format(test_acc))

            self.train_metrics.reset()
            self.test_metrics.reset()
        
    def save_model(self,save_name:str):
        torch.save(self.model.state_dict())

    def draw_loss(self):
        df = pd.DataFrame(dict(
            epochs = [k+1 for k in range(self.epochs)],
            train_loss_value = self.training_loss,
            test_loss_value = self.testing_loss,
        ))
        fig = px.line(df, x="epochs", y=["train_loss_value","test_loss_value"], title="Evolution of loss throughout epochs")
        fig.show()


def trainer_test():
    device = torch.device("cpu")

    epochs = 10
    learning_rate = 0.001
    
    leNet5 = get_model("LeNet5")

    metrics = MetricCollection([
            MulticlassAccuracy(num_classes=10)
        ]).to(device)
         
    criterion = nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(leNet5.parameters(), lr=learning_rate)

    transform = Compose([
        Resize((32,32)),
        ToTensor()
        
    ])
    train_data = torchvision.datasets.MNIST("data/MNIST", train=True, download=False, transform=transform)
    test_data = torchvision.datasets.MNIST("data/MNIST", train=False, download=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    trainer = Trainer(
        epochs,
        leNet5,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        metrics,
        device
    )
    trainer.run()
    trainer.draw_loss()