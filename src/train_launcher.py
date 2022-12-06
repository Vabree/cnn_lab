import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Resize

from Trainer import Trainer
from models.model_inventory import get_model
from data.data_inventory import get_datas
from utils.deepLearning.losses.loss_inventory import get_loss
from utils.deepLearning.metrics.metric_inventory import get_metric
from utils.common.transform_list import get_transform

def train_launcher(config: dict):
    torch.seed()

    #Setup the device where the training will be performed
    device_name = config['device']
    device = torch.device(device_name)

    #Load the task 
    task = config['task']

    #Load the config dictionnary
    model_config = config['model']
    dataset_config = config['dataset']
    hyperparameters_config = config['hyperparameters']

    model = get_model(model_config['architecture'], dataset_config['channel_number'], model_config['class_number'])

    #Load the loss function
    loss_name = config['criterion']
    criterion = get_loss(loss_name)

    #Load the optimizer
    learning_rate = hyperparameters_config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #Loard the metrics bundle
    
    if task == "image_classification":
        metrics_dict = config['image_classification_metrics']
        metrics = get_metric(task, metrics_dict,  model_config['class_number'])
    elif task == "semseg":
        metrics_dict = config['semseg_metrics']
        metrics = get_metric(config['task'], metrics_dict,  model_config['class_number'])
    else:
        raise NameError("task not available")

    #Load the transform 
    transform_config = dataset_config['transforms']
    train_transform = get_transform(transform_config['train'])
    test_transform = get_transform(transform_config['test'])
    target_transform = get_transform(transform_config['target'])

    #Load the dataset
    train_data, test_data = get_datas(dataset_config, train_transform, test_transform, target_transform)

    #Transform the dataset into dataloader
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=hyperparameters_config['batch_size'], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=hyperparameters_config['batch_size'], shuffle=True)

    #Create the trainer object with all elements loaded above
    trainer = Trainer(
        hyperparameters_config['epochs'],
        model,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        metrics,
        device
    )

    #Start the training
    trainer.run()

    #Save the model
    if config['saving_name'] != "None":
        trainer.save_model(config['saving_name'])

    #Draw losses curves
    trainer.draw_loss()