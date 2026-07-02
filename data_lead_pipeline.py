"""
Пайплайн Data Lead: приведение датасета к единому формату + сбор hard negatives + train/val/test split
 
Что нужно перед запуском:
    pip install datasets librosa soundfile pandas numpy tqdm
 
Что делает скрипт:
    1. Загружает drone-audio-detection-samples (HF), сохраняет каждый файл как .wav
       с единой частотой дискретизации и единой длиной клипа.
    2. Обрабатывает вашу папку с hard negatives (газонокосилка, дрель, вертолёт) тем же способом.
    3. Делает train/val/test split.
    4. Сохраняет итоговую таблицу метаданных metadata.csv.
 
Результат:
    processed_data/train/*.wav
    processed_data/val/*.wav
    processed_data/test/*.wav
    metadata.csv
"""
 
import io
import os
 
# --- Перенаправляем кэш Hugging Face на диск D (до импорта datasets!) ---
# Поменяйте путь на любую удобную папку на диске D, если хотите другое название.
os.environ["HF_HOME"] = "D:\\hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:\\hf_cache\\datasets"
 
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio
 
# ---------- НАСТРОЙКИ (поменяйте под себя, если нужно) ----------
 
TARGET_SR = 16000          # единая частота дискретизации
TARGET_DURATION = 4.0      # единая длина клипа в секундах
OUTPUT_DIR = "processed_data"
METADATA_PATH = "metadata.csv"
 
# Путь к вашей папке с hard negatives.
# Ожидается такая структура (создайте сами):
#   hard_negatives/mower/*.wav или *.mp3
#   hard_negatives/drill/*.wav или *.mp3
#   hard_negatives/helicopter/*.wav или *.mp3
HARD_NEGATIVES_DIR = "hard_negatives"
 
# Доли на train/val/test
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
 
# ------------------------------------------------------------------
 
 
def fix_length(audio, sr, target_duration):
    """Обрезает или дополняет (тишиной) аудио до нужной длины."""
    target_len = int(sr * target_duration)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    return audio
 
 
def save_processed(audio, sr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, sr)
 
 
def process_hf_dataset():
    """Шаг 1: скачивает HF-датасет и приводит все клипы к единому формату."""
    print("Загружаю датасет с Hugging Face...")
    dataset = load_dataset("geronimobasso/drone-audio-detection-samples")
    split = dataset["train"]
 
    # Отключаем автоматическое декодирование аудио (оно требует torchcodec + FFmpeg).
    # Вместо этого получаем сырые байты файла и декодируем их сами через soundfile.
    split = split.cast_column("audio", Audio(decode=False))
 
    # Выводим, что означают классы 0/1 — ОБЯЗАТЕЛЬНО проверьте это в консоли
    label_feature = split.features["label"]
    print("Расшифровка классов:", label_feature)
 
    records = []
    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp_hf")
 
    for i, item in enumerate(tqdm(split, desc="Обработка HF датасета")):
        audio_bytes = item["audio"]["bytes"]
        label_id = item["label"]
        label_name = label_feature.int2str(label_id)  # например '0' / '1'
 
        # декодируем аудио вручную из сырых байтов через soundfile (без torchcodec/FFmpeg)
        audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_array.ndim > 1:  # если вдруг стерео — сводим в моно
            audio_array = audio_array.mean(axis=1)
 
        # ресэмплинг к целевой частоте
        if sr != TARGET_SR:
            audio_array = librosa.resample(
                np.asarray(audio_array, dtype=np.float32), orig_sr=sr, target_sr=TARGET_SR
            )
 
        audio_array = fix_length(audio_array, TARGET_SR, TARGET_DURATION)
 
        filename = f"hf_{i:06d}.wav"
        out_path = os.path.join(tmp_dir, filename)
        save_processed(audio_array, TARGET_SR, out_path)
 
        records.append({
            "filepath": out_path,
            "class": label_name,
            "source_dataset": "HF",
            "duration_sec": TARGET_DURATION,
            "sample_rate": TARGET_SR,
        })
 
    return pd.DataFrame(records)
 
 
def process_hard_negatives():
    """Шаг 2: обрабатывает вашу папку с газонокосилкой/дрелью/вертолётом."""
    records = []
    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp_hard_neg")
 
    if not os.path.isdir(HARD_NEGATIVES_DIR):
        print(f"Папка {HARD_NEGATIVES_DIR} не найдена — пропускаю hard negatives.")
        print("Создайте её и положите звуки в подпапки mower/ drill/ helicopter/")
        return pd.DataFrame(records)
 
    for class_name in os.listdir(HARD_NEGATIVES_DIR):
        class_dir = os.path.join(HARD_NEGATIVES_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
 
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".wav", ".mp3", ".flac"))]
        for i, fname in enumerate(tqdm(files, desc=f"Обработка hard negative: {class_name}")):
            path = os.path.join(class_dir, fname)
            audio_array, sr = librosa.load(path, sr=None)
 
            if sr != TARGET_SR:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
 
            audio_array = fix_length(audio_array, TARGET_SR, TARGET_DURATION)
 
            out_filename = f"hardneg_{class_name}_{i:04d}.wav"
            out_path = os.path.join(tmp_dir, out_filename)
            save_processed(audio_array, TARGET_SR, out_path)
 
            records.append({
                "filepath": out_path,
                "class": f"hard_negative_{class_name}",
                "source_dataset": "custom",
                "duration_sec": TARGET_DURATION,
                "sample_rate": TARGET_SR,
            })
 
    return pd.DataFrame(records)
 
 
def make_split(df):
    """Шаг 3: train/val/test split со случайным перемешиванием внутри каждого класса."""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # перемешать
 
    splits = []
    for class_name, group in df.groupby("class"):
        n = len(group)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
 
        group = group.reset_index(drop=True)
        group.loc[:n_train - 1, "split"] = "train"
        group.loc[n_train:n_train + n_val - 1, "split"] = "val"
        group.loc[n_train + n_val:, "split"] = "test"
        splits.append(group)
 
    return pd.concat(splits).reset_index(drop=True)
 
 
def move_to_final_location(df):
    """Перемещает файлы из временных папок в processed_data/train|val|test/."""
    new_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Раскладываю по train/val/test"):
        old_path = row["filepath"]
        split = row["split"]
        filename = os.path.basename(old_path)
        new_path = os.path.join(OUTPUT_DIR, split, filename)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.replace(old_path, new_path)
        new_paths.append(new_path)
 
    df["filepath"] = new_paths
    return df
 
 
def main():
    df_hf = process_hf_dataset()
    df_hard_neg = process_hard_negatives()
 
    df_all = pd.concat([df_hf, df_hard_neg], ignore_index=True)
 
    df_all = make_split(df_all)
    df_all = move_to_final_location(df_all)
 
    df_all.to_csv(METADATA_PATH, index=False)
 
    print("\nГотово! Итоговая сводка:")
    print(df_all.groupby(["class", "split"]).size())
    print(f"\nВсего файлов: {len(df_all)}")
    print(f"Метаданные сохранены в: {METADATA_PATH}")
    print(f"Обработанные файлы лежат в: {OUTPUT_DIR}/")
 
 
if __name__ == "__main__":
    main()