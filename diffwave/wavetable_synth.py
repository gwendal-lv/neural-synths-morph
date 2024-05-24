"""
Differentiable wavetable synthesis component.
"""
import torch
from torch import nn
import numpy as np
from utils import *
import soundfile as sf
import matplotlib.pyplot as plt
from core import upsample


def wavetable_osc(wavetable, freqs, sr):
    """
    General wavetable synthesis oscillator.
    TODO implement antialiasing... ICASSP22 suggest a filter for high f0
        But: which f0 to use? The median (for each batch item) from freqs? Or require an F0 as an arg?

    :param freqs: F0s extracted using CREPE on time frames, upsampled at sr
    """
    # Wavetable indexes computation: does not need to be differentiable (freqs estimation was not)
    freqs = freqs.squeeze()  # Shape N_minibatch x L_audio
    L_wt = wavetable.shape[0]  # Length of the waveform
    increment = (L_wt / sr) * freqs
    index = torch.cumsum(increment, dim=1) - increment[0]
    # Issue with a simple modulo here: index seems to get so close to 512 that floor returns 512.0
    #    Then index_low becomes 512 and indexing the wavetable crashes
    index = torch.remainder(index, L_wt)  # index % L_wt
    # So: use an eps ; if too close to the max, force the index to 0. Cannot be done later, because we need a proper
    #     index value in order to compute the alpha
    index[((L_wt - index) < 1e-5)] = 0.0

    # uses linear interpolation implementation
    index_low, index_high = torch.floor(index.clone()), torch.ceil(index.clone())  # TODO explain: why clone?
    alpha = index - index_low
    index_low, index_high = index_low.long(), index_high.long()

    # The modulo in index_high ensures that the interpolation also works when going back to the wave's start sample
    index_high = index_high % L_wt
    output = wavetable[index_low] + alpha * (wavetable[index_high] - wavetable[index_low])
        
    return output


def generate_wavetable(length, f, cycle=1, phase=0):
    """
    Generate a wavetable of specified length using 
    function f(x) where x is phase.
    Period of f is assumed to be 2 pi.
    """
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(cycle * 2 * np.pi * i / length + 2 * phase * np.pi)
    return torch.tensor(wavetable)


class WavetableSynth(nn.Module):
    def __init__(self, wavetables=None, n_wavetables=64, wavetable_len=512, sr=16000, duration_secs=3, block_size=160):
        """

        :param wavetables:
        :param n_wavetables:
        :param wavetable_len:  Default 512. diffwave ICASSP22 paper says OK for the lowest 20Hz F0 at 16kHz; but
                                sampling the fundamental wave at 1/sr corresponds to 16kHz / 512 = 31.25Hz
        :param sr:
        :param duration_secs:
        :param block_size:
        """
        super(WavetableSynth, self).__init__()
        self.sr, self.block_size = sr, block_size

        if wavetables is None:
            self.wavetables = []
            for _ in range(n_wavetables):
                cur = nn.Parameter(torch.empty(wavetable_len).normal_(mean=0, std=0.01))
                self.wavetables.append(cur)

            self.wavetables = nn.ParameterList(self.wavetables)

            for idx, wt in enumerate(self.wavetables):
                # TODO use fixed wavetable ONLY IF REQUIRED - default behavior should be end-to-end learning
                # following the paper, initialize f0-f3 wavetables and disable backprop FIXME not what it says...
                # Regarding phase: ICASSP22 paper states:
                #     "We found phase-locking wavetables to start and end at 0 deteriorated performance."
                #     in contrast to DDSP, locks harmonic components at 0 phase.
                #     BUT: FIXME why a random phase for the first constant harmonics???
                # The strange concat from original repo definitely breaks periodicity;
                #     a notch in output waveform (supposedly sinusoidal) is even visible.
                # The ICASSP22 paper indicated to add an extra identical sample at the (length 512 -> 513);
                #     we handle this by managing the indexing properly, the result is the same.
                if idx == 0:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=1)  # , phase=random.uniform(0, 1))
                    #wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 1:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=2)  # , phase=random.uniform(0, 1))
                    #wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 2:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=3)  # , phase=random.uniform(0, 1))
                    #wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 3:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=4)  # , phase=random.uniform(0, 1))
                    #wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                else:
                    #wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)  # TODO why here also???
                    wt.requires_grad = True
        else:
            self.wavetables = wavetables

    def forward(self, pitch, envelope, attention):
        """
        Expected inputs' shape: N_minibatch x L_raw_audio x N_wavetables

        :param pitch:
        :param envelope: A(t) in the ICASSP22 paper
        :param attention: Time-varying attention weights (already softmaxed and upsampled) for all waveforms.
                          Corresponds to c_i(t) in the paper.
        :return:
        """
        output_waveform_lst = []
        # This could be improved (parallelized) by using a 2D matrix as wavetables. Not sure if it's worth it...
        # Generate constant-envelope waveforms from the wavetables
        for wt_idx in range(len(self.wavetables)):
            wt = self.wavetables[wt_idx]
            if wt_idx not in [0, 1, 2, 3]:
                # FIXME does compression!!! in the original paper, some wavetable have amplitudes > 1
                pass  # wt = nn.Tanh()(wt)  # ensure wavetable range is between [-1, 1]  TODO why now? TRY REMOVE
            waveform = wavetable_osc(wt, pitch, self.sr)
            output_waveform_lst.append(waveform)
        output_waveform = torch.stack(output_waveform_lst, dim=2)
        # Then apply the attentions (local envelopes for each waveform)
        output_waveform = output_waveform * attention  # actually multiple waveforms at this point
        output_waveform = torch.sum(output_waveform, dim=2, keepdim=True)
        output_waveform *= envelope
        return output_waveform


if __name__ == "__main__":
    pass # TODO debug/test anti-aliasing here....


