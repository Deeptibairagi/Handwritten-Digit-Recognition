from Imports import *

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))

])


full_train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)


train_size = 50000
val_size = 10000

train_data, val_data = random_split(full_train_data, [train_size, val_size])
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
