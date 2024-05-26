"""
Diff-WTS model. Main adapted from https://github.com/acids-ircam/ddsp_pytorch.
"""

import torch
import torch.nn as nn
from torchvision.transforms import Resize

import nnAudio.features.mel

from core import harmonic_synth
from core import mlp, gru, scale_magnitudes, remove_above_nyquist, upsample
from core import amp_to_impulse_response, fft_convolve, multiscale_fft, safe_log
from wavetable_synth import WavetableSynth


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


def config_to_model_kwargs(config):
    return {
        'hidden_size': config["model"]["hidden_size"], 'n_harmonic': config["model"]["n_harmonic"],
        'n_bands': config["model"]["n_bands"],
        'sampling_rate': config["common"]["sampling_rate"], 'block_size': config["common"]["block_size"],
        'n_wavetables': config["model"]["n_wavetables"], 'n_wt_pure_harmonics': config['model']['n_wt_pure_harmonics'],
        'mode': config["model"]["synth_mode"], 'duration_secs': config["common"]["duration_secs"],
        'n_mfcc': config["model"]["n_mfcc"], 'upsampling_mode': config["model"]["upsampling_mode"]
    }


class DDSP_WTS(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size,
                 n_wavetables, n_wt_pure_harmonics=0,
                 mode="wavetable", duration_secs=3,
                 n_mfcc=30, use_reverb=False, upsampling_mode="nearest"):
        super().__init__()
        self.hidden_size, self.synth_mode, self.duration_secs, self.n_mfcc = hidden_size, mode, duration_secs, n_mfcc
        self.upsampling_mode = upsampling_mode  # Use to upsample pitches, loudness and partial's amplitudes
        assert self.synth_mode in ["harmonic", "wavetable"]
        # Parameters that can be saved and restored, but not trained
        #    Will be moved to the GPU
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # Pre-processing
        self.mean_loudness, self.std_loudness = -39.74668743704927, 54.19612404969509  # NSynth train set
        #self.mean_loudness, self.std_loudness = -28.654367014678332, 59.86895554249753  # TODO USE Dexed train set
        self.compute_MFCC = nnAudio.features.mel.MFCC(sr=sampling_rate, n_mfcc=n_mfcc)

        # Original DDSP: "normalization layer with learnable shift and scale parameters"
        self.layer_norm = nn.LayerNorm(self.n_mfcc)
        self.gru_mfcc = nn.GRU(self.n_mfcc, self.hidden_size, batch_first=True)
        self.mlp_mfcc = nn.Linear(self.hidden_size, 16)  # 16 (latent variables per frame) = size of a z(t)

        self.decoder_in_mlps = nn.ModuleList([mlp(1, self.hidden_size, 3),
                                              mlp(1, self.hidden_size, 3),
                                              mlp(16, self.hidden_size, 3)])
        self.decoder_gru = gru(3, self.hidden_size)  # TODO use raw GRU... not that method
        self.out_mlp = mlp(self.hidden_size * 4, self.hidden_size, 3)  # 4 tokens after the skip-connection

        # Synthesis modules
        if self.synth_mode == "harmonic":
            self.partials_projection = nn.Linear(hidden_size, n_harmonic + 1)
            self.wts = None
        elif self.synth_mode == "wavetable":
            self.partials_projection = nn.Linear(hidden_size, n_wavetables + 1)
            self.wts = WavetableSynth(n_wavetables=n_wavetables, sr=sampling_rate, n_pure_harmonics=n_wt_pure_harmonics)
        else:
            raise ValueError(self.synth_mode)
        self.noise_projection = nn.Linear(hidden_size, n_bands)
        self.reverb = Reverb(sampling_rate, sampling_rate) if use_reverb else None

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))


    def forward(self, mfcc, pitch, loudness, Z=None):
        loudness = (loudness - self.mean_loudness) / self.std_loudness
        pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()

        # - - - Encode mfcc first ; or bypass if Z was provided directly - - -
        if Z is None:
            mfcc = torch.transpose(mfcc, 1, 2)
            # Shape after transpose: N x L_MFCC x n_MFCC
            #     where L_MFCC is the number of MFCC frames != L_frames (number of audio frames (tokens))
            # use layer norm instead of trainable norm, not much difference found
            mfcc = self.layer_norm(mfcc)  # could use train dataset's stats...
            Z = self.gru_mfcc(mfcc)[0]  # output [1] would be the last hidden state for each GRU layer - we discard it
            Z = self.mlp_mfcc(Z)  # After this: Z shape is N x L_MFCC x 16
        else:
            assert mfcc is None, "Either MFCC or optional Z should be provided."
        # use image resize to align dimensions, ddsp also do this... TODO CHECK this: seems OK but can't find the source
        # FIXME use a variable factor for resize ; 100 only works with the default config 16kHz block 160
        Z_resampled = Resize(size=(self.duration_secs * 100, 16))(Z)  # After this: Z shape is N x L_frames x 16

        # - - - Decoder - - -
        pitch_hidden = self.decoder_in_mlps[0](pitch)
        loudness_hidden = self.decoder_in_mlps[1](loudness)
        Z_decoder_hidden = self.decoder_in_mlps[2](Z_resampled)
        decoder_hidden = torch.cat([pitch_hidden, loudness_hidden, Z_decoder_hidden], -1)
        # Why bypass the GRU? Skip-connection? DDSP paper indicates a skip-co from F0; then, page 16:
        # "we concatenate the GRU outputs with outputs of f(t) and l(t) MLPs (in the channel dimension)";
        #     not Z's MLP output
        # Concat in channel dim: tokens' size increases (4 times hidden dim)
        decoder_gru_outputs = self.decoder_gru(decoder_hidden)[0]  # Shape N x L_frames x hidden_size
        decoder_hidden = torch.cat([decoder_gru_outputs, decoder_hidden], -1) # Shape N x L_frames x 4*hidden_size
        decoder_hidden = self.out_mlp(decoder_hidden) # Shape N x L_frames x hidden_size


        # - - - harmonic/wavetable signal synthesis - - -
        #   (also used to compute wavetable's components amplitudes, although not truly 'harmonic')
        #   handled slightly differently for DDSP and DiffWave
        harm_param = self.partials_projection(decoder_hidden)
        envelope, amplitudes = harm_param[..., :1], harm_param[..., 1:]
        if self.synth_mode == "harmonic":  # ddsp synthesizer
            envelope, amplitudes = scale_magnitudes(envelope), scale_magnitudes(amplitudes)
            amplitudes /= amplitudes.sum(-1, keepdim=True)  # TODO why this for DDSP??? try remove this...
            amplitudes *= envelope
            # "Removes" higher harmonics for high pitches;
            #    still a 0.0001 factor for aliased components... for consistent backprop?
            # this can't prevent aliasing for wavetable synthesis... (only for the first fixed harmonic waves)
            amplitudes = remove_above_nyquist(amplitudes, pitch, self.sampling_rate)
        elif self.synth_mode == "wavetable":  # diff-wave-synth synthesizer
            # ICASSP22 paper: "A(n) and ci (n) are constrained positive via a sigmoid."
            #    but for ci(n): probably shouldn't apply a sigmoid then a softmax....
            envelope = torch.sigmoid(envelope) + 1e-7  # same as DDSP, for stability
            # softmax for amplitudes. Amplitudes' shape: N_minibatch x L_frames x N_wavetables
            amplitudes = torch.softmax(amplitudes, dim=2)
        # upsample synthesis parameters to audio rate
        pitch = upsample(pitch, self.block_size, self.upsampling_mode)
        envelope = upsample(envelope, self.block_size, self.upsampling_mode)
        amplitudes = upsample(amplitudes, self.block_size, self.upsampling_mode)
        # Then generate the harmonic signal  TODO give output shape
        if self.synth_mode == "harmonic":
            # Envelope has been applied already
            harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)  # shape N_minibatch x L_raw_audio x 1
        elif self.synth_mode == "wavetable":
            harmonic = self.wts(pitch, envelope, amplitudes)  # not truly harmonic...
        else:
            raise ValueError(self.synth_mode)


        # - - - noise signal synthesis - - -
        noise_param = scale_magnitudes(self.noise_projection(decoder_hidden) - 5)
        impulse = amp_to_impulse_response(noise_param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1
        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)


        # - - - Sum signals and add optional reverb - - -
        signal = harmonic + noise
        if self.reverb is not None:
            signal = self.reverb(signal)
        return signal, Z


def MSS_loss(audio_output, audio_target, scales, overlap, add_log_loss):
    ori_stft = multiscale_fft(audio_target.squeeze(), scales, overlap)
    rec_stft = multiscale_fft(audio_output.squeeze(), scales, overlap)
    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        loss += lin_loss
        if add_log_loss:
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss += log_loss
    return loss
