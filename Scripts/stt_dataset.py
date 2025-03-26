import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperDataset(Dataset):
    def __init__(self, path_csv, processor, sample_rate=16000):
        """
        data_list: Lista de diccionarios con {"audio_path": str, "text": str}
        processor: WhisperProcessor para tokenizar texto
        sample_rate: Frecuencia de muestreo objetivo (16 kHz para Whisper)
        """
        self.data_list = pd.read_csv(path_csv)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # Cargar el audio
        audio, sr = torchaudio.load(item["path_x"],sr=16000)
        if sr != self.sample_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = transform(audio)

        audio = audio.mean(dim=0)  # Convertir a mono si es estéreo

        # Tokenizar la transcripción
        input_features = self.processor(audio.numpy(), sampling_rate=self.sample_rate, return_tensors="pt").input_features[0]
        labels = self.processor.tokenizer(item["sentence"]).input_ids

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels)
        }


model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
dataset = WhisperDataset(path_csv=r'F:\common_voice\Proyecto\Scripts\spanish_stt.csv',processor=WhisperProcessor)
dataset[0]