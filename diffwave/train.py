"""
Train script.
"""

import datetime
from pathlib import Path

import numpy as np
import torch

import yaml
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloader import get_data_loader
from model import DDSP_WTS, MSS_loss
import plots


def train_model(config):

    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)   issue with matmult and the current PyTorch and CUDA versions

    # general parameters
    device = config['model']['device']
    sr = config["common"]["sampling_rate"]
    duration_secs = config["common"]["duration_secs"]
    batch_size = config["train"]["batch_size"]
    scales = config["train"]["scales"]
    overlap = config["train"]["overlap"]
    epochs = config["train"]["epochs"]

    assert config['train']['loss'] in ['lin', 'lin+log']  # TODO move to dedicated loss class
    add_log_loss = (config['train']['loss']  == 'lin+log')


    # TODO use a config to kwargs method here
    model = DDSP_WTS(
        hidden_size=config["model"]["hidden_size"], n_harmonic=config["model"]["n_harmonic"],
        n_bands=config["model"]["n_bands"], sampling_rate=sr, block_size=config["common"]["block_size"],
        n_wavetables=config["model"]["n_wavetables"], mode=config["model"]["synth_mode"], duration_secs=duration_secs,
        n_mfcc=config["model"]["n_mfcc"], upsampling_mode=config["model"]["upsampling_mode"]
    )
    model.to(device)

    # full batches only? maybe not be required if batch norm is not used (seems to be layer norm only)
    train_dl = get_data_loader(config, mode="train", batch_size=batch_size)
    valid_dl = get_data_loader(config, mode="valid", batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["start_lr"])
    gamma = np.exp((np.log(config["train"]["stop_lr"]) - np.log(config["train"]["start_lr"]))
                   / ((config["train"]["epochs"] * len(train_dl) / 100) - 1))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    # define tensorboard writer
    root_logs_dir = Path(config["train"]["root_logs_dir"])
    assert root_logs_dir.exists() and root_logs_dir.is_dir()
    # log dir should be shared with other synths - run name uses wt or hm for wavetable or harmonic ddsp
    synth_mode_short = {'wavetable': 'wt', 'harmonic': 'hm'}[config["model"]["synth_mode"]]
    # Add the current time to any model name that starts with "debug"
    if config['model']['name'].lower().startswith('debug'):
        config['model']['name'] += '_' + datetime.datetime.now().strftime('%m%d-%H%M')  #('%Y%m%d-%H%M%S')
    log_dir = root_logs_dir.joinpath(f"ddsp_{synth_mode_short}_{config['model']['name']}")
    print(f"===== Model name: '{config['model']['name']}' =====.\nLogging into {str(log_dir)}")

    train_logger = SummaryWriter(str(log_dir.joinpath('train')))
    valid_logger = SummaryWriter(str(log_dir.joinpath('valid')))

    with open(log_dir.joinpath('config.yaml'), 'w') as f:
        yaml.dump(config, f)
    step_index = 0
    for epoch in tqdm(range(1, epochs + 1)):
        for audio_target, loudness, pitch in train_dl:
            audio_target, loudness, pitch = audio_target.to(device), loudness.to(device), pitch.to(device)
            mfcc = model.compute_MFCC(audio_target)
            audio_output, Z = model(mfcc, pitch, loudness)

            loss = MSS_loss(audio_output, audio_target, scales, overlap, add_log_loss)
            train_logger.add_scalar(f"MSSloss/{config['train']['loss']}", loss.item(), global_step=step_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_index += 1

            # Scheduler step
            if step_index % 100 == 0:
                lr_scheduler.step()
                train_logger.add_scalar(f"LR", lr_scheduler.get_last_lr(), global_step=step_index)

        # - - - Validation (at each epoch) - - -
        with torch.no_grad():
            losses = list()
            for audio_target, loudness, pitch in valid_dl:
                audio_target, loudness, pitch = audio_target.to(device), loudness.to(device), pitch.to(device)
                mfcc = model.compute_MFCC(audio_target)
                audio_output, Z = model(mfcc, pitch, loudness)
                losses.append(MSS_loss(audio_output, audio_target, scales, overlap, add_log_loss).item())
            valid_logger.add_scalar(f"MSSloss/{config['train']['loss']}", np.mean(losses), global_step=step_index)

            # - - - plots (pas à toutes les epochs) and save the model - - -
            if epoch % 10 == 0:  # Epoch starts at 1 (not 0)
                # Plot the wavetable, also compute stats and log those
                if model.synth_mode == 'wavetable':
                    wavetables = np.asarray([wt.clone().detach().cpu().numpy() for wt in model.wts.wavetables])
                    fig, axes, average_rms, average_max_amplitude = plots.wavetables(wavetables)
                    # TODO proper logging... AND remember that a tanh is applied to learned WTs
                    train_logger.add_scalar("wavetable/RMS", average_rms, global_step=step_index)
                    train_logger.add_scalar("wavetable/max", average_max_amplitude, global_step=step_index)
                    train_logger.add_figure("wavetables_plot", fig, close=True, global_step=step_index)

                # Small models (a few MBs)
                torch.save(model.state_dict(), log_dir.joinpath("model.pt"))


if __name__ == "__main__":
    with open("config.yaml", 'r') as stream:
        _config = yaml.safe_load(stream)
    train_model(_config)
