from model_loading_28x28 import *

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_28x28.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device used is: {device}")
model_28x28.to(device)