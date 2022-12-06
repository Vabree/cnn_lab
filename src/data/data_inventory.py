import torchvision 



def get_datas(config:dict, train_transform, test_transform, target_transform):
    #Load the dataset
    if config['type'] == "MNIST":
        train_data = torchvision.datasets.MNIST(config["train_path"], train=True, download=False, transform=train_transform)
        test_data = torchvision.datasets.MNIST(config["test_path"], train=False, download=False, transform=test_transform)
    elif config['type'] == "cityscapes":

        train_data = torchvision.datasets.Cityscapes(config["train_path"], split='train', mode='fine',
                                        target_type='semantic', transform=train_transform, target_transform=target_transform)
        test_data = torchvision.datasets.Cityscapes(config["test_path"], split='test', mode='fine',
                                        target_type='semantic', transform=test_transform, target_transform=target_transform)
    else:
        raise NameError("dataset_type not available")

    return train_data, test_data