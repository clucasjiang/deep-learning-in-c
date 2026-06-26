import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path
from model import NeuralNetwork
from save_params import export_c

batch_size = 64
epochs = 20
script_dir = Path(__file__).resolve().parent
data_root = script_dir.parent / "mnist_dataset"

training_data = datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

training_dataloader = DataLoader(training_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_number, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_number % 100 == 0:
            loss, current = loss.item(), (batch_number + 1) * len(x)
            print(f"loss: {loss:>7f}. [{current:>5d}/{size:>5d}]")
        
            
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")
    train(training_dataloader, model, loss_fn, optimizer)
    
torch.save(
    {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    script_dir / "mnist_model.pth",
)
export_c(model, script_dir / "mlp_weights.c")

# Epoch = 5 Acc = 72.31%
# Epoch = 6 Acc = 77.20%
# Epoch = 8 Acc = 80.79%
# Epoch = 10 Acc = 83.08%
# Epoch = 15 Acc = 87.79%
# Epoch = 20 Acc = 89.05%
