"""
Perform morphings for given set of start and end audio samples.
"""
import json
import shutil
from pathlib import Path
import time

import librosa
import torch
import numpy as np
import yaml
from tqdm import tqdm
import soundfile as sf

from model import DDSP_WTS, config_to_model_kwargs

def morph(model_dir: Path, dataset_dir: Path, n_steps=9, device='cuda:0'):
    """

    :param model_dir: Model directory with the config.yaml files and model weights.
    :param dataset_dir: Directory of the validation or test dataset: contains the audio, loudness, pitch subfolders,
                        and the morph_sequences.json file
    """
    # Reload the config, then the model
    print(f"Loading model {model_dir.parent.name}/{model_dir.name}...")
    with open(model_dir.joinpath('config.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)
    sr = config["common"]["sampling_rate"]

    model = DDSP_WTS(**config_to_model_kwargs(config))
    model.load_state_dict(torch.load(model_dir.joinpath("model.pt")))
    model.eval()
    torch.set_grad_enabled(False)
    model.to(device)

    # Now process the whole dataset
    # The .json file should be a list of dicts where each dict describes a sequence, e.g. :
    # {"seq_index": 1498, "audio_start": "044165_pitch056vel075_var000.wav", "audio_end": "077637_pitch056vel075_var000.wav"}
    with open(dataset_dir.joinpath("morph_sequences.json"), 'r') as f:
        morph_sequences = json.load(f)

    morph_save_dirs = {name: model_dir.joinpath(f'Morph{name}') for name in ('MFCC', 'Z')}
    for k in list(morph_save_dirs.keys()):
        morph_save_dirs[k].mkdir(parents=False, exist_ok=True)
        morph_save_dirs[k] = morph_save_dirs[k].joinpath(f'interp{n_steps}_{dataset_dir.name}')
        if morph_save_dirs[k].exists():
            shutil.rmtree(morph_save_dirs[k])
        morph_save_dirs[k].mkdir(parents=False, exist_ok=False)

    for seq_idx, morph_seq in enumerate(tqdm(morph_sequences)):
        assert seq_idx == morph_seq['seq_index']
        audio_names = [morph_seq['audio_start'], morph_seq['audio_end']]
        # Retrieve pre-computed pitch and loudness (don't use the torch dataset for this... to ensure audio UIDs)
        audio_start_end, loudness_start_end, pitch_start_end = list(), list(), list()
        for a in audio_names:
            audio_start_end.append(librosa.load(dataset_dir.joinpath(f'audio/{a}'), sr=sr)[0])
            loudness_start_end.append(np.load(dataset_dir.joinpath(f"loudness/{a.replace('.wav', '_loudness.npy')}")))
            pitch_start_end.append(np.load(dataset_dir.joinpath(f"pitch/{a.replace('.wav', '_pitch.npy')}")))
        # The model will add the 3rd singleton dim
        audio_start_end, loudness_start_end, pitch_start_end = \
            (torch.from_numpy(np.vstack(x)).to(device) for x in (audio_start_end, loudness_start_end, pitch_start_end))

        # Reconstruct start and end
        mfcc_start_end = model.compute_MFCC(audio_start_end)
        audio_out_start_end, Z_start_end = model(mfcc_start_end, pitch_start_end, loudness_start_end)

        # Interp of latent values
        alpha = torch.linspace(0.0, 1.0, n_steps).to(device)
        alpha = alpha[1:-1].view(n_steps - 2, 1)
        pitch_interp = (1.0 - alpha) * pitch_start_end[0:1, :] + alpha * pitch_start_end[1:2, :]
        loudness_interp = (1.0 - alpha) * loudness_start_end[0:1, :] + alpha * loudness_start_end[1:2, :]
        alpha = alpha.view(n_steps - 2, 1, 1)
        mfcc_interp = (1.0 - alpha) * mfcc_start_end[0:1, :, :] + alpha * mfcc_start_end[1:2, :, :]
        Z_interp = (1.0 - alpha) * Z_start_end[0:1, :, :] + alpha * Z_start_end[1:2, :, :]

        # MFCC-based and Z-based resynthesis
        audio_interp_MFCC, _ = model(mfcc_interp, pitch_interp, loudness_interp)
        audio_interp_Z, _ = model(None, pitch_interp, loudness_interp, Z=Z_interp)

        # Save results
        for interp_method, audio_interp in zip(("MFCC", "Z"), (audio_interp_MFCC, audio_interp_Z)):
            audio = torch.cat((audio_out_start_end[0:1, ...], audio_interp, audio_out_start_end[1:2, ...]), dim=0)
            seq_dir = morph_save_dirs[interp_method].joinpath(f'{seq_idx:05d}')
            seq_dir.mkdir(parents=False, exist_ok=False)
            for i in range(n_steps):
                sf.write(seq_dir.joinpath(f'audio_step{i:02d}.wav'), audio[i, :, 0].cpu().numpy(), sr)


if __name__ == "__main__":
    morph(
        Path("/media/gwendal/Data/Logs/neural-synths-morph/final/ddsp_wt_20wt5pureharm"),
        Path("/media/gwendal/Data/Datasets/Dexed_split/test/"),
        9
    )

