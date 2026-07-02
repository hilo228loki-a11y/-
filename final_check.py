import torch
import torch.nn as nn
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. Описание архитектуры (копия той, что в train.py)
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

# 2. Ваш загрузчик данных (копия из dataset_loader.py)
class DroneAudioDataset(Dataset):
    def __init__(self, metadata_path, split='test'):
        self.df = pd.read_csv(metadata_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(row['filepath'], sr=16000)
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
        mel = librosa.power_to_db(mel, ref=np.max)
        label = 1 if str(row['class']) == '1' else 0
        return torch.FloatTensor(mel).unsqueeze(0), label

# 3. Основной блок проверки
if __name__ == "__main__":
    # Загружаем модель
    model = DroneCNN()
    try:
        model.load_state_dict(torch.load("drone_model.pth"))
        model.eval()
    except FileNotFoundError:
        print("Ошибка: файл drone_model.pth не найден. Сначала запустите обучение!")
        exit()

    test_dataset = DroneAudioDataset('metadata.csv', split='test')
    total = len(test_dataset)
    correct = 0

    print(f"Начинаю тест на {total} файлах...")

    with torch.no_grad():
        for i, (spec, label) in enumerate(test_dataset):
            output = model(spec.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            if predicted.item() == label:
                correct += 1
            if i % 500 == 0:
                print(f"Обработано {i} файлов...")

    accuracy = (correct / total) * 100
    print("-" * 30)
    print(f"Финальная точность модели: {accuracy:.2f}%")