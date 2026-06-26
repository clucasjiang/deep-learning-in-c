import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path
from model import NeuralNetwork

batch_size = 64
script_dir = Path(__file__).resolve().parent
data_root = script_dir.parent / "mnist_dataset"

test_data = datasets.MNIST(
    root=data_root,
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(script_dir / "mnist_model.pth", map_location=device))
model.eval()

def test(dataloader, model):
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            for (i, pred_label) in enumerate(pred.argmax(1)):
                if pred_label.item() == y[i].item():
                    correct += 1
    return correct / len(dataloader.dataset)
            
accuracy = test(test_dataloader, model) * 100
print(f"Accuracy: {accuracy}%")
