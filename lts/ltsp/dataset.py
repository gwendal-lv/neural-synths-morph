
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class CQTDataset(Dataset):
    def __init__(self, root_dir: Union[Path, str], cqt_rel_path="LTS_CQT"):
        """
        Dataset of precomputed constant-Q transforms generated from ../create_dataset.py

        Replaces the TensorDataset class from the original repo (all audio files were loaded into RAM -
        not compatible with (reasonably) large datasets).
        """
        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), str(self.root_dir.resolve())
        self.audio_dir = self.root_dir / 'audio'
        assert self.audio_dir.exists(), str(self.audio_dir.resolve())
        self.cqt_dir = self.root_dir / cqt_rel_path
        assert self.cqt_dir.exists(), str(self.cqt_dir.resolve())

        self._audio_paths = sorted([p for p in self.audio_dir.glob('*.wav')])
        self._CQT_paths = sorted([p for p in self.cqt_dir.glob('*.npy')])
        # Check that audio and CQT correspond....
        for i, (audio_path, cqt_path) in enumerate(zip(self._audio_paths, self._CQT_paths)):
            assert audio_path.stem == cqt_path.stem, \
                f"Mismatch between audio and CQT file names ({audio_path.stem} and {cqt_path.stem}) for index {i}"

    def __getitem__(self, i):
        cqt = np.load(self._CQT_paths[i])
        return torch.Tensor(cqt)

    def __len__(self):
        return len(self._CQT_paths)

    def get_audio(self, i: int):
        return librosa.load(self._audio_paths[i], sr=None)
