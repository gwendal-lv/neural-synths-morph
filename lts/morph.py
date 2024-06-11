import argparse, configparser
import json
from pathlib import Path
from datetime import datetime
import shutil

import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

import ltsp.model

# TODO time measurements also here
def morph(n_steps=9, device='cpu'):
    torch.set_grad_enabled(False)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint', required=True)
    args = parser.parse_args()
    print(args.checkpoint)
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent.parent.parent
    # Load config, in a parent dir of the checkpoint
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(run_dir / 'config.ini')
    sampling_rate, hop_length, bins_per_octave, num_octaves, n_bins, n_iter, cqt_bit_depth, dtype = \
        ltsp.model.process_audio_config(config)

    # Create model and load the weights
    lts_model = ltsp.model.VAE(n_bins, config['VAE'].getint('n_units'), config['VAE'].getint('latent_dim')).to(device)
    checkpoint = torch.load(checkpoint_path)
    lts_model.load_state_dict(checkpoint)
    lts_model.eval()

    # Load test data / list of files (ordered start/end points for sequences)
    test_data_path = Path(config['dataset'].get('test_dataset'))
    with open(test_data_path / 'morph_sequences.json', 'r') as f:
        morph_sequences = json.load(f)  # List of dicts with 'seq_index' int, and 'audio_start' and 'audio_end' paths

    # Create the directory for storing all sequences
    morphs_base_dir = run_dir.joinpath(f'interp{n_steps}_{test_data_path.name}')
    if morphs_base_dir.exists():
        shutil.rmtree(morphs_base_dir)
    morphs_base_dir.mkdir(parents=False, exist_ok=False)
    print(f"Computing morphing.... storage: {morphs_base_dir}")
    # Then process each sequence, one by one (and measure computation times immediately)
    analysis_times, synthesis_times = list(), list()  # Analysis times for start+end samples ; synthesis for 1 step
    for seq_idx, seq in enumerate(tqdm(morph_sequences)):
        # Create seq dir
        assert seq['seq_index'] == seq_idx
        seq_dir = morphs_base_dir.joinpath(f'{seq_idx:05d}')
        seq_dir.mkdir(parents=False, exist_ok=False)
        # Load audio files
        _, original_sr = librosa.load(test_data_path / f"audio/{seq['audio_start']}", sr=None)  # retrieve the orig SR
        t0 = datetime.now()
        start_end_CQTs = torch.from_numpy(np.stack([
            ltsp.model.compute_CQT_from_file(f, sampling_rate, hop_length, bins_per_octave, n_bins, cqt_bit_depth)
            for f in [test_data_path / f"audio/{seq['audio_start']}", test_data_path / f"audio/{seq['audio_end']}"]
        ], axis=0)).to(lts_model.dtype)
        # Encode start/end only, then reshape the flattened output
        z_mu, z_logvar = lts_model.encode(start_end_CQTs.view(-1, n_bins))
        analysis_times.append((datetime.now() - t0).total_seconds())
        z_start_end = z_mu.reshape(2, start_end_CQTs.shape[1], lts_model.latent_dim)
        alpha = (torch.arange(0.0, n_steps) / (n_steps - 1.0)).view(n_steps, 1, 1)
        z_interp = ((1.0 - alpha) * z_start_end[0:1, :, :]) + (alpha * z_start_end[1:2, :, :])
        # Decode everything without flattening ("de-batching") z_interp
        #    with integrated phase reconstruction
        for i in range(n_steps):
            t0 = datetime.now()
            audio = lts_model.generate_audio_from_latent(
                z_interp[i, :, :], 1,
                sr=sampling_rate, n_iter=n_iter, hop_length=hop_length, bins_per_octave=bins_per_octave, dtype=dtype
            )[0]
            synthesis_times.append((datetime.now() - t0).total_seconds())
            # Resample and save audio
            audio = librosa.resample(y=audio, orig_sr=sampling_rate, target_sr=original_sr)
            sf.write(seq_dir / f'audio_step{i:02d}.wav', audio, original_sr)
    # print analysis / synthesis time stats
    analysis_times, synthesis_times = np.asarray(analysis_times) * 1000.0, np.asarray(synthesis_times) * 1000.0
    print(f"analysis_times:   mean={analysis_times.mean():.1f}ms   std={analysis_times.std():.1f}ms")
    print(f"synthesis_times:   mean={synthesis_times.mean():.1f}ms   std={synthesis_times.std():.1f}ms")


if __name__ == "__main__":
    morph()
