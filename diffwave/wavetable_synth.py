"""
Differentiable wavetable synthesis component.
"""
import torch
from torch import nn
import numpy as np
from utils import *
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
from core import upsample
import random


def wavetable_osc(wavetable, freq, sr):
    """
    General wavetable synthesis oscillator.
    """
    freq = freq.squeeze()
    increment = freq / sr * wavetable.shape[0]
    index = torch.cumsum(increment, dim=1) - increment[0]
    index = index % wavetable.shape[0]

    # uses linear interpolation implementation
    index_low = torch.floor(index.clone())
    index_high = torch.ceil(index.clone())
    alpha = index - index_low
    index_low = index_low.long()
    index_high = index_high.long()

    # FIXME error in the original implementation:
    #     operator(): block: [2875,0,0], thread: [0,0,0]
    #     Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
    # Pour débugguer: faire check manual avant que CUDA ne crashe...
    #   TODO remove these checks after debug
    assert index_low.shape == index_high.shape
    assert np.all(0 <= index_low.detach().cpu().numpy())
    assert np.all(index_low.detach().cpu().numpy() <= wavetable.shape[0])
    assert np.all(0 <= index_high.detach().cpu().numpy())
    assert np.all(index_high.detach().cpu().numpy() <= wavetable.shape[0])
    output = wavetable[index_low] + alpha * (wavetable[index_high % wavetable.shape[0]] - wavetable[index_low])
        
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
    def __init__(self,
                 wavetables=None,
                 n_wavetables=64,
                 wavetable_len=512,
                 sr=16000,
                 duration_secs=3,
                 block_size=160):
        """

        :param wavetables:
        :param n_wavetables:
        :param wavetable_len:  Default 512 (diffwave ICASSP22 paper) OK for the lowest 20Hz F0 at 16kHz
        :param sr:
        :param duration_secs:
        :param block_size:
        """
        super(WavetableSynth, self).__init__()
        if wavetables is None: 
            self.wavetables = []
            for _ in range(n_wavetables):
                cur = nn.Parameter(torch.empty(wavetable_len).normal_(mean=0, std=0.01))
                self.wavetables.append(cur)

            self.wavetables = nn.ParameterList(self.wavetables)

            for idx, wt in enumerate(self.wavetables):
                # following the paper, initialize f0-f3 wavetables and disable backprop FIXME not what it says...
                # Regarding phase: ICASSP22 paper states:
                #     "We found phase-locking wavetables to start and end at 0 deteriorated performance."
                #     in contrast to DDSP, locks harmonic components at 0 phase.
                #     BUT: FIXME why a random phase for the first constant harmonics???
                if idx == 0:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=1, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 1:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=2, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 2:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=3, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 3:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=4, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                else:
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = True
            
            self.attention = nn.Parameter(torch.randn(n_wavetables, 100 * duration_secs))

        else:
            self.wavetables = wavetables
            self.attention = nn.Parameter(torch.randn(n_wavetables, 100 * duration_secs))
        
        self.sr = sr
        self.block_size = block_size
        self.attention_softmax = nn.Softmax(dim=0)

    def forward(self, pitch, amplitude):        
        output_waveform_lst = []
        for wt_idx in range(len(self.wavetables)):
            wt = self.wavetables[wt_idx]
            if wt_idx not in [0, 1, 2, 3]:
                wt = nn.Tanh()(wt)  # ensure wavetable range is between [-1, 1]  TODO why now?
            waveform = wavetable_osc(wt, pitch, self.sr)
            output_waveform_lst.append(waveform)

        # apply attention TODO explain why attention is not the output of a NN.... only amplitudes ???
        attention = self.attention_softmax(self.attention)
        attention_upsample = upsample(attention.unsqueeze(-1), self.block_size).squeeze()

        output_waveform = torch.stack(output_waveform_lst, dim=1)
        output_waveform = output_waveform * attention_upsample
        output_waveform_after = torch.sum(output_waveform, dim=1)
      
        output_waveform_after = output_waveform_after.unsqueeze(-1)
        output_waveform_after = output_waveform_after * amplitude
       
        return output_waveform_after


if __name__ == "__main__":
    # create a sine wavetable and to a simple synthesis test
    wavetable_len = 512
    sr = 16000
    duration = 4
    freq_t = [739.99 for _ in range(sr)] + [523.25 for _ in range(sr)] + [349.23 for _ in range(sr * 2)]
    freq_t = torch.tensor(freq_t)
    freq_t = torch.stack([freq_t, freq_t, freq_t], dim=0)
    sine_wavetable = generate_wavetable(wavetable_len, np.sin)
    wavetable = torch.tensor([sine_wavetable,])
    
    wt_synth = WavetableSynth(wavetables=wavetable, sr=sr, duration_secs=4)
    amplitude_t = torch.ones(sr * duration,)
    amplitude_t = torch.stack([amplitude_t, amplitude_t, amplitude_t], dim=0)
    amplitude_t = amplitude_t.unsqueeze(-1)

    y = wt_synth(freq_t, amplitude_t, duration)
    print(y.shape, 'y')
    plt.plot(y.squeeze()[0].detach().numpy())
    plt.show()
    sf.write('test_3s_v1.wav', y.squeeze()[0].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v2.wav', y.squeeze()[1].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v3.wav', y.squeeze()[2].detach().numpy(), sr, 'PCM_24')



