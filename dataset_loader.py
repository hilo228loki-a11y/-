import torch
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DroneAudioDataset(Dataset):
    def __init__(self, metadata_path, split='train'):
        # Читаем CSV
        self.df = pd.read_csv(metadata_path)
        # Фильтруем по split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Загружаем аудио (используем путь из колонки 'filepath')
        audio, _ = librosa.load(row['filepath'], sr=16000)
        
        # 2. Превращаем в мел-спектрограмму
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # 3. Определяем метку:
        # В вашем CSV в колонке 'class' написано '0' или '1' (для дронов)
        # или 'hard_negative_...' (для фоновых звуков).
        # Значит, дрон — это когда class == '1'
        label = 1 if str(row['class']) == '1' else 0
        
        return torch.FloatTensor(mel).unsqueeze(0), label

# Проверка загрузчика
if __name__ == "__main__":
    dataset = DroneAudioDataset('metadata.csv', split='train')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Всего файлов в наборе: {len(dataset)}")
    specs, labels = next(iter(loader))
    print(f"Размер батча спектрограмм: {specs.shape}")
    print(f"Метки в первом батче: {labels}")