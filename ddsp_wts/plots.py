
import numpy as np
import matplotlib.pyplot as plt

import librosa


def wavetables(_wavetables: np.ndarray, n_fixed_waves=0, colW=5.0, rowH=1.5):
    """
    Plots the given wavetables, and also computes some statistics about the learned (non-constant) wavetables.

    :param _wavetables:
    :param n_fixed_waves: Number of constant wavetables, considered to be the first in _wavetables
    :return: fig, axes, average_rms (constant wts excluded), average_max_amplitude (constant wts excluded)
    """
    assert len(_wavetables.shape) == 2, "2D matrix required"
    assert _wavetables.shape[0] >= 4, "Display is optimized for numerous wavetables"
    n_cols = np.maximum(int(np.floor(np.sqrt(_wavetables.shape[0]) - 1.0)), 2)
    n_rows = int(np.round(np.ceil(_wavetables.shape[0] / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', figsize=(n_cols*colW, n_rows*rowH))
    all_rms, all_max_amplitude = list(), list()
    for i, wt in enumerate(_wavetables):
        # plot the wavetable
        col = i // n_rows
        row = i % n_rows
        axes[row, col].plot(wt, color=('k' if i < n_fixed_waves else 'C0'))
        # and add stats to the figure itself
        rms = np.sqrt(np.mean(np.square(wt)))
        axes[row, col].set_title(f"Wave #{i}   min={wt.min():.2f} rms={rms:.2f} max={wt.max():.2f}")
        if i >= n_fixed_waves:
            all_rms.append(rms)
            all_max_amplitude.append(np.abs(wt).max())
    fig.tight_layout()
    return fig, axes, np.mean(all_rms), np.mean(all_max_amplitude)


def spectrograms(target_audio, reconstructed_audio):
    assert target_audio.shape[0] == reconstructed_audio.shape[0] == 2, "Method will plot 2 audio files only"
    fig, axes = plt.subplots(2, 2, sharex='col', figsize=(5, 5))
    for row in range(2):
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(target_audio[row, :], n_fft=1024)))
        img = librosa.display.specshow(S_db, ax=axes[row, 0], cmap='viridis')
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(reconstructed_audio[row, :], n_fft=1024)))
        img = librosa.display.specshow(S_db, ax=axes[row, 1], cmap='viridis')
    axes[0, 0].set_title("Original audio")
    axes[0, 1].set_title("Reconstructed audio")
    fig.tight_layout()
    return fig, axes
