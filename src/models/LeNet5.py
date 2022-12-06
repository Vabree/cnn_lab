import torch 
import torch.nn as nn 


class BlockCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockCNN,self).__init__()

        self.blockCNN = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 5, stride = 1, padding = 0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2)
        )

    def forward(self,x):
        return self.blockCNN(x)


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5,self).__init__()

        self.encoder = nn.Sequential(
            BlockCNN(in_channels = 1, out_channels = 6),
            BlockCNN(in_channels = 6, out_channels = 16),
            nn.Conv2d(in_channels = 16, out_channels=120, kernel_size = 5, stride = 1, padding = 0),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features = 120, out_features = 84),
            nn.Tanh(),
            nn.Linear(in_features = 84, out_features = n_classes),
        )

    def forward(self,x):
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        logits = self.decoder(feat)
        return logits


if __name__ == "__main__":
    learning_rate = 0.001
    n_classes = 10
    model = LeNet5(n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
