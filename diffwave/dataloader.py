"""
Dataloader files.
"""
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import time
import yaml 
from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(self, config, tr_val="train"):
        self.root = config["dataset"][f"{tr_val}_path"]
        
        self.audio_path = config["dataset"]["audio"]
        self.pitch_path = config["dataset"]["pitch"]
        self.loudness_path = config["dataset"]["loudness"]

        self.audios = sorted(os.listdir(os.path.join(self.root, self.audio_path)))
        self.config = config

    def __getitem__(self, index):
        audio_path = self.audios[index]
        loudness_path = os.path.join(self.root, self.loudness_path, audio_path.replace(".wav", "_loudness.npy"))
        pitch_path = os.path.join(self.root, self.pitch_path, audio_path.replace(".wav", "_pitch.npy"))
        audio_path = os.path.join(self.root, self.audio_path, audio_path)

        y, sr = librosa.load(audio_path, sr=self.config["common"]["sampling_rate"])
        loudness = np.load(loudness_path)
        pitch = np.load(pitch_path)

        return y, loudness, pitch

    def __len__(self):
        return len(self.audios)


def get_data_loader(config, mode="train", batch_size=16, shuffle=True):
    dataloader = DataLoader(dataset=AudioDataset(config, tr_val=mode),
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    ds = AudioDataset(config, tr_val="train")

    mean = 0
    std = 0
    n = 0
    for _, l, _ in tqdm(ds):
        n += 1
        mean += l.mean().item()
        std += l.std().item()

