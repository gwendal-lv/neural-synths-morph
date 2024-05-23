"""
Train script.
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

import yaml
from nnAudio import Spectrogram
from tqdm import tqdm

from core import multiscale_fft, get_scheduler, safe_log
from dataloader import get_data_loader
from model import WTS
# FIXME don't use this anymore... or maybe do if is working with GPU TF disabled ???
#    TODO supposed to easily work with comet
# from tensorboardX import SummaryWriter

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
device = config['model']['device']
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
duration_secs = config["common"]["duration_secs"]
batch_size = config["train"]["batch_size"]
scales = config["train"]["scales"]
overlap = config["train"]["overlap"]
train_lr = config["train"]["start_lr"]
epochs = config["train"]["epochs"]

print(f"""
======================
synth_mode: {config["model"]["synth_mode"]}
======================
sr: {sr}
block_size: {block_size}
duration_secs: {duration_secs}
batch_size: {batch_size}
scales: {scales}
overlap: {overlap}
hidden_size: {config["model"]["hidden_size"]}
n_harmonic: {config["model"]["n_harmonic"]}
n_bands: {config["model"]["n_bands"]}
n_wavetables: {config["model"]["n_wavetables"]}
n_mfcc: {config["model"]["n_mfcc"]}
train_lr: {train_lr}
======================
""")

model = WTS(
    hidden_size=config["model"]["hidden_size"], n_harmonic=config["model"]["n_harmonic"],
    n_bands=config["model"]["n_bands"], sampling_rate=sr,
    block_size=block_size, n_wavetables=config["model"]["n_wavetables"], mode=config["model"]["synth_mode"],
    duration_secs=duration_secs, n_mfcc=config["model"]["n_mfcc"]
)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=train_lr)
spec = Spectrogram.MFCC(sr=sr, n_mfcc=config["model"]["n_mfcc"])

# both values are pre-computed from the train set 
mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509  # FIXME load from dataset's statistics

# TODO full batches only? maybe not required if batch norm is not used (seems to be layer norm only)
train_dl = get_data_loader(config, mode="train", batch_size=batch_size)
valid_dl = get_data_loader(config, mode="valid", batch_size=batch_size)  # TODO use this...

# for now the scheduler is not used
schedule = get_scheduler(
    len(train_dl),
    config["train"]["start_lr"],
    config["train"]["stop_lr"],
    config["train"]["decay_over"],
)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/' + current_time +'/train'

#train_summary_writer = SummaryWriter(train_log_dir)

step_index = 0
for epoch in tqdm(range(1, epochs + 1)):
    for audio_target, loudness, pitch in train_dl:

        # TODO this should be done in the model itself (easier to use...)
        pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
        loudness = (loudness - mean_loudness) / std_loudness

        # Compute MFCC on the CPU FIXME should be done on the GPU (debug this...) or pre-computed
        mfcc = spec(audio_target).to(device)
        # TODO move audio to the device
        audio_target, loudness, pitch = audio_target.to(device), loudness.to(device), pitch.to(device)

        output = model(mfcc, pitch, loudness)

        # TODO compute also ori_stft on the GPU?
        ori_stft = multiscale_fft(audio_target.squeeze(), scales, overlap)
        rec_stft = multiscale_fft(output.squeeze(), scales, overlap)

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            loss += lin_loss
            # TODO OPTIONAL LOG LOSS
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss += log_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        # TODO log this into comet? or maybe just discard summary writer.... ?
        #train_summary_writer.add_scalar('loss', loss.item(), global_step=idx)
        if step_index % 10 == 0:
            print(f"loss = {loss}")
        if step_index % 500 == 0:
            torch.save(model.state_dict(), "model.pt")
        
        step_index += 1  # TODO rename step idx

    # TODO implement some kind of eval...