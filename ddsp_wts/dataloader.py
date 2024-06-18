"""
Dataloader files.
"""
import json
import os
from pathlib import Path
import time
import pickle

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml 
from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(self, config, tr_val="train"):
        self.root = config["dataset"][f"{tr_val}_path"]
        self.root_dir = Path(self.root)
        
        self.audio_path = config["dataset"]["audio"]
        self.pitch_path = config["dataset"]["pitch"]
        self.loudness_path = config["dataset"]["loudness"]
        if config["dataset"]["timbre"] is not None and config["dataset"]["timbre"] != "":
            self.timbre_dir = self.root_dir / config["dataset"]["timbre"]
            with open(self.root_dir / 'normalized_timbre_features_names.json', 'r') as f:
                available_timbre_features = list(json.load(f))
            timbre_features_to_use = config["dataset"]["timbre_features"].split(';')
            # make bool mask to select a limited set of features
            self.timbre_features_bool_mask = torch.zeros((len(available_timbre_features), ), dtype=torch.bool)
            for f_name in timbre_features_to_use:
                assert f_name in available_timbre_features
                idx = available_timbre_features.index(f_name)
                self.timbre_features_bool_mask[idx] = True
            print(f"[AudioDataset {tr_val}] {torch.count_nonzero(self.timbre_features_bool_mask).item()} "
                  f"timbre features used")
        else:
            self.timbre_dir, self.timbre_features_bool_mask = None, None

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

        if self.timbre_dir is not None:
            timbre_path = self.timbre_dir / (Path(audio_path).stem + '.torch.pickle')
            with open(timbre_path, 'rb') as f:
                timbre = pickle.load(f)
            timbre = timbre[self.timbre_features_bool_mask]
        else:
            timbre = torch.empty((0, ))

        return y, loudness, pitch, timbre

    def __len__(self):
        return len(self.audios)

    @property
    def n_timbre_features(self):
        if self.timbre_features_bool_mask is None:
            return 0
        else:
            return torch.count_nonzero(self.timbre_features_bool_mask).item()


def get_data_loader(config, mode="train", batch_size=16, shuffle=True, drop_last=False):
    dataloader = DataLoader(
        dataset=AudioDataset(config, tr_val=mode), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
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

