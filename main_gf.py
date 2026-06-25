import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

train_dir = "/home/jovyan/work/datasets/6a39ed934d7b489daf5f80a4-momodel/train"

im_size = 224
batch_size = 32
num_classes = 4
epochs = 15
lr = 1e-3
val_ratio = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WeatherCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 26 * 26, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_loaders():
    tf = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
    ])

    full_set = datasets.ImageFolder(train_dir, transform=tf)
    print('class_to_idx:', full_set.class_to_idx)

    n_val = int(len(full_set) * val_ratio)
    n_train = len(full_set) - n_val
    train_set, val_set = random_split(
        full_set, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False)
    return train_loader, val_loader


def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def train():
    train_loader, val_loader = build_loaders()
    model = WeatherCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f'Epoch {epoch}/{epochs}  '
              f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
              f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}')

    os.makedirs('/home/jovyan/work/results', exist_ok=True)
    torch.save(model.state_dict(),
               '/home/jovyan/work/results/model_sample.pth')


if __name__ == '__main__':
    train()
