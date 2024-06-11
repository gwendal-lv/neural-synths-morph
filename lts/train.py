# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, argparse, time
from pathlib import Path

import numpy as np
import librosa
import configparser
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import tensorboardX

from ltsp.model import VAE, loss_function, process_audio_config
import ltsp.dataset



def train(seed=20240607):
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./default.ini', help='path to the config file')
    args = parser.parse_args()

    # Get configs
    config_path = args.config
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        config.read(config_path)
    except FileNotFoundError:
        print('Config File Not Found at {}'.format(config_path))
        sys.exit()
    # Import audio configs
    sampling_rate, hop_length, bins_per_octave, num_octaves, n_bins, n_iter, cqt_bit_depth, dtype = \
        process_audio_config(config)
    # Training configs
    n_epochs = config['training'].getint('epochs')
    learning_rate = config['training'].getfloat('learning_rate')
    batch_size = config['training'].getint('batch_size')
    checkpoint_interval = config['training'].getint('checkpoint_interval')
    plot_interval = config['training'].getint('plot_interval')
    n_workers = config['training'].getint('n_workers')
    if sys.gettrace() is not None:  # Debugging check - will stop working w/ PyCharm 2023.3
        n_workers = 0

    # Datasets and dataloaders
    training_dataset = ltsp.dataset.CQTDataset(config['dataset'].get('datapath'))
    validation_dataset = ltsp.dataset.CQTDataset(config['dataset'].get('valid_dataset'))
    print(f"Datasets lengths: training {len(training_dataset)}, validation {len(validation_dataset)}")
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Workspace directory - where the training runs will be stored
    if config['dataset'].get('workspace') != '':
        workspace_dir = Path(config['dataset'].get('workspace'))
        assert workspace_dir.exists(), str(workspace_dir.resolve())
    else:
        workspace_dir = Path(__file__).parent / 'workspace'
        workspace_dir.mkdir(parents=False, exist_ok=True)
    # Create run directory in workspace dir
    run_number = config['dataset'].getint('run_number')
    run_id = run_number
    while True:
        try:
            run_name = 'run{:03d}'.format(run_id)
            run_dir = workspace_dir / run_name
            os.makedirs(run_dir)
            break
        except OSError:
            if run_dir.is_dir():
                run_id = run_id + 1
                continue
    print("Run will be stored in: {}".format(run_dir))
    logger = tensorboardX.SummaryWriter(logdir=str(run_dir))

    # Model configs
    latent_dim = config['VAE'].getint('latent_dim')
    n_units = config['VAE'].getint('n_units')
    kl_beta = config['VAE'].getfloat('kl_beta')
    device = config['VAE'].get('device')

    start_time = time.time()
    config['extra']['start'] = time.asctime(time.localtime(start_time))

    device = torch.device(device)
    device_name = torch.cuda.get_device_name()
    print('Device: {}'.format(device_name))
    config['VAE']['device_name'] = device_name

    config_path = run_dir / 'config.ini'
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    # Train
    model_dir = run_dir / "model"
    checkpoint_dir = model_dir / 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Neural Network
    model = VAE(n_bins, n_units, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_index = 0
    for epoch in tqdm(range(n_epochs), desc='Epoch', leave=False, position=0):
        model.train()
        train_loss = 0

        for i, x_cqt in enumerate(tqdm(training_dataloader, desc="Minibatch", leave=False, position=1)):

            x_cqt = x_cqt.to(device)
            optimizer.zero_grad()
            flattened_out, mu, logvar = model(x_cqt)
            recons_loss, KLD_with_beta = loss_function(flattened_out, x_cqt, mu, logvar, kl_beta, n_bins)
            total_loss = recons_loss + KLD_with_beta
            total_loss.backward()
            train_loss += total_loss.item()
            optimizer.step()

            logger.add_scalar("AudioMSE/train", recons_loss, global_step=step_index)
            logger.add_scalar("KLD/train", KLD_with_beta, global_step=step_index)
            logger.add_scalar("loss/train", total_loss, global_step=step_index)

            # Validation every X steps
            if step_index % 100 == 0 and step_index > 0:
                with torch.no_grad():
                    valid_recons_loss, valid_KLD_with_beta = 0.0, 0.0
                    for j, x_cqt in enumerate(validation_dataloader):
                        x_cqt = x_cqt.to(device)
                        flattened_out, mu, logvar = model(x_cqt)
                        recons_loss, KLD_with_beta = loss_function(flattened_out, x_cqt, mu, logvar, kl_beta, n_bins)
                        valid_recons_loss += recons_loss / len(validation_dataloader)
                        valid_KLD_with_beta += KLD_with_beta / len(validation_dataloader)
                    # Log averaged validation values
                    logger.add_scalar("AudioMSE/valid", valid_recons_loss, global_step=step_index)
                    logger.add_scalar("KLD/valid", valid_KLD_with_beta, global_step=step_index)
                    logger.add_scalar("loss/valid", valid_recons_loss + valid_KLD_with_beta, global_step=step_index)

            step_index += 1

        # Model save, plots, etc...
        with torch.no_grad():
            if (epoch % checkpoint_interval == 0 and epoch > 0) or (epoch == n_epochs - 1):
                torch.save(model.state_dict(), checkpoint_dir / f'model_epoch{epoch:05d}')

            if (epoch % plot_interval == 0) or (epoch == n_epochs - 1):
                # Generate a few audio files
                # Retrieve original audio
                indices = [(step_index + i) % len(validation_dataset) for i in range(4)]
                audio_original = [validation_dataset.get_audio(i) for i in indices]
                sr_original = [a[1] for a in audio_original]
                assert len(set(sr_original)) == 1, "All original files must have the same sampling rate"
                sr_original = sr_original[0]
                audio_original = [a[0] for a in audio_original]
                x_cqt = torch.stack([validation_dataset[i] for i in indices], dim=0).to(device)
                # Reconstruct audio
                audio_out, z_flattened = model.generate_audio_from_CQT(
                    x_cqt,
                    sr=sampling_rate, n_iter=n_iter, hop_length=hop_length, bins_per_octave=bins_per_octave, dtype=dtype
                )
                if sampling_rate != sr_original:
                    audio_out = [librosa.resample(y=a, orig_sr=sampling_rate, target_sr=sr_original) for a in audio_out]
                # Build sequence of original+reconstruct ; save to tensorboard
                audio_cat = np.concatenate(sum([[a1, a2] for a1, a2 in zip(audio_original, audio_out)], []))
                logger.add_audio("audio", audio_cat, sample_rate=sr_original, global_step=step_index)


    with open(config_path, 'w') as configfile:
        config.write(configfile)



if __name__ == "__main__":
    train()
