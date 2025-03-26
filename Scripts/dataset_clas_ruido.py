import os
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset
import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np

path_dataset = r'Scripts\data_proc\audios_proc'

class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, n_mels=128, transform=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Lista de clases (carpetas)
        self.files = []

        # Recorre las carpetas y almacena rutas y etiquetas
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith(".mp3"):  # Solo archivos .mp3
                    self.files.append((os.path.join(class_path, file_name), class_idx))

        # Transformación de Mel-spectrograma
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            #hop_length=512,
            normalized=True
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path, label = self.files[idx]

        # Cargar audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample si la tasa de muestreo no es la deseada
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)


        if waveform.shape[1] < 10*sr:
            pad = 10*sr - waveform.shape[1]
            last_dim_padding = (0,pad)
            waveform = torch.nn.functional.pad(waveform,last_dim_padding)

        # Convertir a Mel-spectrograma
        mel_spec = self.mel_transform(waveform)

        # Aplicar transformaciones adicionales (opcional)
        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, label

# Ejemplo de uso

if __name__ == "__main__":
    dataset = AudioDataset(audios_dir=path_dataset)
    print(len(dataset))
    mel_spec, label = dataset[4000]

    print("Mel-Spectrogram shape:", mel_spec.shape)  # (1, n_mels, tiempo)
    print("Label:", label)