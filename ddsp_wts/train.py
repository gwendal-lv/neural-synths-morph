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
import soundfile as sf

from dataloader import get_data_loader
from model import DDSP_WTS, MSS_loss, AR_latent_loss, config_to_model_kwargs
import plots


def train_model(config):

    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)   issue with matmult and the current PyTorch and CUDA versions

    # general parameters
    device = config['model']['device']
    sr = config["common"]["sampling_rate"]
    batch_size = config["train"]["batch_size"]
    scales = config["train"]["scales"]
    overlap = config["train"]["overlap"]
    epochs = config["train"]["epochs"]

    assert config['train']['loss'] in ['lin', 'lin+log']  # TODO move to dedicated loss class
    add_log_loss = (config['train']['loss']  == 'lin+log')


    model = DDSP_WTS(**config_to_model_kwargs(config))
    model.to(device)

    # full batches only? maybe not be required if batch norm is not used (seems to be layer norm only)
    train_dl = get_data_loader(config, mode="train", batch_size=batch_size, drop_last=True)
    valid_dl = get_data_loader(config, mode="valid", batch_size=batch_size, shuffle=False)

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
    for epoch in tqdm(range(1, epochs + 1), desc='Epoch', position=0):  # FIXME check leave (True or False)
        for batch in tqdm(train_dl, desc="Minibatch", leave=False, position=1):
            audio_target, loudness, pitch, timbre = (item.to(device) for item in batch)
            mfcc = model.compute_MFCC(audio_target)
            audio_output, Z = model(mfcc, pitch, loudness)

            loss_mss = MSS_loss(audio_output, audio_target, scales, overlap, add_log_loss)
            loss_AR = AR_latent_loss(Z, timbre)
            loss_total = loss_mss + loss_AR
            train_logger.add_scalar(f"MSSloss/{config['train']['loss']}", loss_mss.item(), global_step=step_index)
            if isinstance(loss_AR, torch.Tensor):
                train_logger.add_scalar("ARloss", loss_AR.item(), global_step=step_index)
            train_logger.add_scalar("TotalLoss", loss_total.item(), global_step=step_index)

            # TODO try implement extra wavetable loss: minimize cross-correlation because the wavetables?
            #   (otherwise, multiple waveforms tend to become the same).
            # Risk is that wavetables degenerate to low-correlation noise...

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            step_index += 1

            # Scheduler step
            if step_index % 100 == 0:
                lr_scheduler.step()
                train_logger.add_scalar(f"LR", lr_scheduler.get_last_lr(), global_step=step_index)

        # - - - Validation (at each epoch) - - -
        with torch.no_grad():
            valid_audio_target, valid_audio_output = None, None
            losses = {f"MSSloss/{config['train']['loss']}": list(), "ARloss": list(), 'TotalLoss': list()}
            for batch in valid_dl:
                audio_target, loudness, pitch, timbre = (item.to(device) for item in batch)
                mfcc = model.compute_MFCC(audio_target)
                audio_output, Z = model(mfcc, pitch, loudness)
                loss_mss = MSS_loss(audio_output, audio_target, scales, overlap, add_log_loss)
                loss_AR = AR_latent_loss(Z, timbre)
                loss_total = loss_mss + loss_AR
                losses[f"MSSloss/{config['train']['loss']}"].append(loss_mss.item())
                if isinstance(loss_AR, torch.Tensor):
                    losses['ARloss'].append(loss_AR.item())
                losses['TotalLoss'].append(loss_total.item())
                # Backup of the first minibatch only; may be plotted/saved just after this
                if valid_audio_target is None:
                    valid_audio_target = audio_target.cpu().numpy()
                    valid_audio_output = audio_output.cpu().squeeze(dim=2).numpy()
            for k, l_array in losses.items():
                if len(l_array) > 0:  # Don't log disabled null AR loss
                    valid_logger.add_scalar(k, np.mean(l_array), global_step=step_index)

            # - - - plots (pas à toutes les epochs) and save the model - - -
            if epoch % config['train']['plot_period_epochs'] == 0:  # Epoch starts at 1 (not 0)
                # Save audio (first items from the saved minibatch), directly into the logs folder
                #    we concatenate a few GT/reconstructed samples and save everything into a single file
                concat_audio = list()
                for i in range(10):
                    concat_audio += [valid_audio_target[i, :], valid_audio_output[i, :]]
                concat_audio = np.concatenate(concat_audio)
                sf.write(log_dir.joinpath(f'audio_epoch{epoch:03d}.flac'), concat_audio, sr)
                # Plot 2 original and reconstructed spectrograms (or mel-specs?)
                fig, axes = plots.spectrograms(valid_audio_target[0:2, :], valid_audio_output[0:2, :])
                valid_logger.add_figure("audio_recons", fig, close=True, global_step=step_index)
                # Plot the wavetable, also compute stats and log those
                if model.synth_mode == 'wavetable':
                    wavetables = np.asarray([wt.clone().detach().cpu().numpy() for wt in model.wts.wavetables])
                    fig, axes, average_rms, average_max_amplitude = plots.wavetables(
                        wavetables, n_fixed_waves=config['model']['n_wt_pure_harmonics'])
                    train_logger.add_scalar("wavetable/RMS", average_rms, global_step=step_index)
                    train_logger.add_scalar("wavetable/max", average_max_amplitude, global_step=step_index)
                    train_logger.add_figure("wavetables_plot", fig, close=True, global_step=step_index)

                # Small models (a few MBs)
                torch.save(model.state_dict(), log_dir.joinpath("model.pt"))
    torch.save(model.state_dict(), log_dir.joinpath("model.pt"))

if __name__ == "__main__":
    with open("config.yaml", 'r') as stream:
        _config = yaml.safe_load(stream)
    train_model(_config)
