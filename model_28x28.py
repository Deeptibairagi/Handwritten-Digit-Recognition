from Data_loader_28x28 import *


class Handwritten(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(

        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(

        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )


        self.fc1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64*7*7, 128),
        nn.ReLU()

        )

        self.fc2=nn.Sequential(
        nn.Linear(128, 10)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x