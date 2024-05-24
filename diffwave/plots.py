
import numpy as np
import matplotlib.pyplot as plt

import torch


def wavetables(_wavetables: np.ndarray, n_fixed_waves=4):
    """
    Plots the given wavetables, and also computes some statistics about the learned (non-constant) wavetables.

    :param _wavetables:
    :param n_fixed_waves: Number of constant wavetables, considered to be the first in _wavetables
    :return: fig, axes, average_rms (constant wts excluded), average_max_amplitude (constant wts excluded)
    """
    assert len(_wavetables.shape) == 2, "2D matrix required"
    n_cols = _wavetables.shape[0] // 5
    n_rows = int(np.round(np.ceil(_wavetables.shape[0] / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', figsize=(n_cols*5.0, n_rows*1.5))
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


