import os
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset
import torch
path_dataset = r'F:\common_voice\Proyecto\Scripts\data_proc\audios_proc'

class AudioMultiDatasetWAVE(Dataset):
    def __init__(self, audios_dir, sample_rate=16000, n_mels=128, transform=None):
        
        self.path_audios = audios_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.transform = transform
        self.files = [(os.path.join(audios_dir,i),i.split('_')[2]) for i in os.listdir(audios_dir) if i.endswith('mp3') ]
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=512,
            normalized=True
        )

    def __len__(self):

        return  len(self.files)
    
    def __getitem__(self, idx):
        audio_path, label = self.files[idx]

        nums_lang = {'ar':0,'es':1,'fr':2}
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
        #mel_spec = self.mel_transform(waveform)

        # Aplicar transformaciones adicionales (opcional)
        if self.transform:
            mel_spec = self.transform(mel_spec)

        return waveform, nums_lang[label]
    

class AudioMultiDataset(Dataset):
    def __init__(self, audios_dir, sample_rate=16000, n_mels=128, transform=None):
        
        self.path_audios = audios_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.transform = transform
        self.files = [(os.path.join(audios_dir,i),i.split('_')[2]) for i in os.listdir(audios_dir) if i.endswith('mp3') ]
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=512,
            normalized=True
        )

    def __len__(self):

        return  len(self.files)
    
    def __getitem__(self, idx):
        audio_path, label = self.files[idx]

        nums_lang = {'ar':0,'es':1,'fr':2}
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

        return mel_spec, nums_lang[label]

dataset = AudioMultiDataset(audios_dir=path_dataset)
print(len(dataset))
waves, label = dataset[10]

print("Mel-Spectrogram shape:", waves.shape)  # (1, n_mels, tiempo)
print("Label:", label)