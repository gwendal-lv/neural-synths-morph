"""
Differentiable wavetable synthesis component.
"""

import numpy as np
import torch
from torch import nn


def wavetable_osc(wavetable_padded, freqs, sr, fir_h=None):
    """
    General wavetable synthesis oscillator.
    Antialiasing... ICASSP22 suggest a "filter for high f0"

    :param wavetable_padded: Must be reflection- (mirror-) padded if fir_h is given. shape 1 x 1 x L_wavetable
    :param freqs: F0s extracted using CREPE on time frames, upsampled at sr (shape N_minibatch x L_raw_audio x 1)
    :param fir_h: Very basic antialiasing filter; should properly work for low F0s only.
    :returns output waveforms, shape N_minibatch x L_raw_audio
    """
    # The filtering is here (not done outside this method) because it should eventually depend on the local f0.
    #    Computationally: does not change the cost; filtering is applied once per minibatch for each wave from
    #    the wavetable.
    # (torchaudio.functional.convolve not available yet in our current PyTorch version)
    if fir_h is not None:
        assert len(fir_h.shape) == len(wavetable_padded.shape) == 3, "3D tensors expected for torch conv"
        assert tuple(fir_h.shape[0:2]) == tuple(wavetable_padded.shape[0:2]) == (1, 1)
        wavetable = torch.nn.functional.conv1d(wavetable_padded, fir_h).squeeze()  # Won't work for high f0
    else:
        wavetable = wavetable_padded
    # A true solution to prevent aliasing may be to integrate aliasing quantification as a backproped loss....
    # Or: generate time-varying filters (much more expensive to apply/compute), cut-off depending on local F0?

    # Wavetable indexes computation: does not need to be differentiable (freqs estimation was not)
    freqs = torch.squeeze(freqs, dim=2)  # After squeeze: Shape N_minibatch x L_audio
    L_wt = wavetable.shape[0]  # Length of the waveform
    increment = (L_wt / sr) * freqs
    index = torch.cumsum(increment, dim=1) - increment[0]  # "float index" which is some kind of accumulating phase
    # Issue with a simple modulo here: index seems to get so close to 512 that floor returns 512.0
    #    Then index_low becomes 512 and indexing the wavetable crashes
    index = torch.remainder(index, L_wt)  # not sure that torch's remainder is better than the modulo operator
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


def windowed_sinc_low_pass_filter(n_taps, nu_c, window=torch.hamming_window):
    h = 2 * nu_c * torch.sinc(2 * nu_c * torch.arange(-(n_taps - 1) // 2, 1 + (n_taps - 1) // 2))
    return h * window(n_taps, periodic=False)


class WavetableSynth(nn.Module):
    def __init__(self, wavetables=None, n_wavetables=10, n_pure_harmonics=0, wavetable_len=512, sr=16000,
                 lowpass_fir_taps=0, lowpass_fir_nu_c=0.1):
        """

        :param wavetables:
        :param n_wavetables:
        :param wavetable_len:  Default 512. diffwave ICASSP22 paper says OK for the lowest 20Hz F0 at 16kHz;
                                sampling the fundamental wave at 1/sr corresponds to 16kHz / 512 = 31.25Hz
        :param sr:
        :param lowpass_fir_taps: Optional filter: 0 deactivates filtering before indexing. TODO Try 21? (31 too expensive...)
        :param lowpass_fir_nu_c: Normalized cut-off frequency in [0.0, 0.5[. The default value is a compromise
                                 that leaves some high-frequency contents for a low F0 and a reasonable amont of
                                 aliasing for a high F0.
        """
        super(WavetableSynth, self).__init__()
        self.sr, self.n_pure_harmonics = sr, n_pure_harmonics

        # Low-pass filtering (before accumulated-phase indexing)
        self.fir_taps, self.fir_nu_c = lowpass_fir_taps, lowpass_fir_nu_c
        if self.fir_taps > 0:
            assert self.fir_taps % 2 == 1,  "Odd number of taps expected (Type I Low-Pass)"
            assert 0.05 <= self.fir_nu_c <= 0.45, "Don't expect too small or too high cut-offs (Nyquist freq = 0.5)"
            h_lowpass = windowed_sinc_low_pass_filter(self.fir_taps, self.fir_nu_c)
            # unsqueeze fir coefficients: will be used as torch 1D conv kernels
            self.fir_h = nn.Parameter(h_lowpass.view(1, 1, self.fir_taps))
            self.fir_h.requires_grad = False
        else:
            self.fir_h = None

        # Cannot use a matrix wavetable (would reduce training times / nb CUDA ops...) because somes lines are
        # learned, other are not
        if wavetables is None:  # Build wavetables (some might be forced to be a purely sinusoidal harmonic)
            # For reproducibility: always init the wavetables with the same random seeds
            self.wavetables = list()
            h_lowpass, n_reflect = windowed_sinc_low_pass_filter(31, 0.1).view(1, 1, 31), 15
            for i in range(n_wavetables):
                rng_cpu = torch.Generator()
                rng_cpu.manual_seed(20240525 + i)
                # TODO check other STDs ?
                wt = torch.normal(mean=torch.zeros((wavetable_len, )), std=0.01, generator=rng_cpu)
                # TODO maybe apply LP filter right after initialization?
                #    to prevent training on white noise
                #wt = torch.cat((wt[-n_reflect:], wt, wt[:n_reflect]), dim=0).view(1, 1, -1)
                #wt = torch.nn.functional.conv1d(wt, h_lowpass).squeeze()
                self.wavetables.append(nn.Parameter(wt))
            self.wavetables = nn.ParameterList(self.wavetables)

            # Initialize pure harmonic wavetables (non-learnable)
            for idx, wt in enumerate(self.wavetables):
                # Regarding phase: ICASSP22 paper states:
                #     "We found phase-locking wavetables to start and end at 0 deteriorated performance."
                #     in contrast to DDSP, locks harmonic components at 0 phase.
                # The paper also indicated to add an extra identical sample at the (length 512 -> 513);
                #     we handle this by managing the indexing properly, the result is the same.
                if idx < self.n_pure_harmonics:
                    # Original code said: "following the paper, initialize f0-f3 wavetables and disable backprop"
                    #  but that's not what the paper says. Observation was: quasi-harmonic WTs appear
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=(idx+1))  # , phase=random.uniform(0, 1))
                    wt.requires_grad = False
                else:  # Randomly initialized; phase will never be locked (unconstrained learning)
                    wt.requires_grad = True

        else:  # Load some pre-computed wavetables
            self.wavetables = wavetables

        # Create reflection-padded versions of the wavetables for filtering - unsqueeze them already for torch conv
        if self.fir_h is not None:
            n_reflect = (self.fir_taps - 1) // 2
            self.wavetables_reflect_padded = nn.ParameterList([
                torch.cat((wt[-n_reflect:], wt, wt[:n_reflect]), dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
                for wt in self.wavetables
            ])
        else:
            self.wavetables_reflect_padded = self.wavetables

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
        # Generate constant-envelope waveforms from the wavetables
        for wt_idx, wt_padded in enumerate(self.wavetables_reflect_padded):
            waveform = wavetable_osc(wt_padded, pitch, self.sr, self.fir_h)  # Filtering may be disabled
            output_waveform_lst.append(waveform)
        output_waveform = torch.stack(output_waveform_lst, dim=2)
        # Then apply the attentions (local envelopes for each waveform)
        output_waveform = output_waveform * attention  # actually multiple waveforms at this point
        output_waveform = torch.sum(output_waveform, dim=2, keepdim=True)
        output_waveform *= envelope
        return output_waveform


