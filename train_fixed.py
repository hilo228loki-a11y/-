import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import DroneAudioDataset

# Архитектура модели
class DroneCNN(nn.Module):
    def __init__(self):
        super(DroneCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(nn.Linear(32 * 16 * 31, 2))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def run_training():
    print("Начинаю обучение...")
    device = torch.device("cpu")
    model = DroneCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = DroneAudioDataset('metadata.csv', split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Обучаем только 500 батчей для проверки (чтобы не ждать часами)
    for i, (specs, labels) in enumerate(train_loader):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Батч {i}, Loss: {loss.item():.4f}")
        if i >= 500: break # Прерываем, как только наберем статистику

    torch.save(model.state_dict(), "drone_model.pth")
    print("Обучение завершено! Файл drone_model.pth создан.")

if __name__ == "__main__":
    run_training()