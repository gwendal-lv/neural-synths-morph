import json
import shutil
from pathlib import Path

import librosa
import torch

import gin
from absl import flags, app
import soundfile as sf
from tqdm import tqdm

try:
    import rave
except:
    import sys, os
    sys.path.append(os.path.abspath('.'))
    import rave


FLAGS = flags.FLAGS

flags.DEFINE_string('run_dir', default=None, help='Path to the run (model directory)', required=True)
flags.DEFINE_string('checkpoint', default=None, required=False,  # If not given, rave.core.search_for_run
                    help='Path (relative to the model directory) to the checkpoint')
flags.DEFINE_string('dataset_dir', default=None, required=True,
                    help="Location of the dataset. This directory must contain the 'morph_sequences.json' file and"
                         " an 'audio' sub-directory with all required audio files.")
flags.DEFINE_integer('n_steps', default=9, required=False, help='Number of interpolation steps')
flags.DEFINE_integer('sr', default=44100, required=False, help='Input/output sampling rate')


def main(argv):
    torch.set_grad_enabled(False)

    in_out_sr, n_steps = FLAGS.sr, FLAGS.n_steps

    # Construct the model (from saved configuration), then load the weights (checkpoint)
    print(f"Loading model from {FLAGS.run_dir}...")
    config_file = rave.core.search_for_config(FLAGS.run_dir)
    assert config_file is not None, f'Config file not found in {FLAGS.run_dir}'
    run_dir = Path(FLAGS.run_dir)
    gin.parse_config_file(config_file)
    # gin handles the configuration (automatically passes the proper ctor args)
    rave_model = rave.RAVE()
    if FLAGS.checkpoint is None:
        checkpoint_path = rave.core.search_for_run(FLAGS.run_dir)
        assert checkpoint_path is not None, f"Checkpoint cannot be automatically found in {run_dir}"
        checkpoint_path = Path(checkpoint_path)
        print(f"Checkpoint automatically selected in {run_dir}")
    else:
        checkpoint_path = run_dir.joinpath(FLAGS.checkpoint)
        assert checkpoint_path.exists(), str(checkpoint_path)
    print(f"Loading checkpoint '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_short_name = checkpoint_path.stem.replace('-epoch=', '')
    rave_model.load_state_dict(checkpoint["state_dict"], strict=False)
    rave_model.eval()

    # Load list of files to be used as tart and end points
    dataset_dir = Path(FLAGS.dataset_dir)
    print(f"Will load audio files from '{dataset_dir}'")
    with open(dataset_dir.joinpath('morph_sequences.json'), 'r') as f:
        morph_seqs = json.load(f)
    # Prepare output dir for morphing
    morphs_base_dir = checkpoint_path.parent.parent.joinpath(f'morph_{checkpoint_short_name}')
    print(f"Sequences of morphed samples will be stored in '{morphs_base_dir}'")
    morphs_base_dir.mkdir(parents=False, exist_ok=True)  # Dir for the model checkpoint
    # Sub dir for the number of interpolation steps (will be erased)
    morphs_base_dir = morphs_base_dir.joinpath(f'interp{n_steps}_{dataset_dir.name}')
    if morphs_base_dir.exists():
        shutil.rmtree(morphs_base_dir)
    morphs_base_dir.mkdir(parents=False, exist_ok=False)

    # Process all required sequences
    for seq_idx, seq_info in enumerate(tqdm(morph_seqs)):
        assert seq_info['seq_index'] == seq_idx
        seq_dir = morphs_base_dir.joinpath(f'{seq_idx:05d}')
        seq_dir.mkdir(parents=False, exist_ok=False)

        audio_start_end = list()
        for a_name in [seq_info['audio_start'], seq_info['audio_end']]:
            x, sr = sf.read(dataset_dir.joinpath(f'audio/{a_name}'))
            assert sr == in_out_sr, f'Found {sr}Hz; expected {in_out_sr}Hz'
            if sr != rave_model.sr:
                x = librosa.resample(y=x, orig_sr=sr, target_sr=rave_model.sr)  # resample to 44.1kHz (RAVE internal)
            audio_start_end.append(torch.from_numpy(x).float().unsqueeze(dim=0))  # unsqueeze 1-ch audio
        # Create batch dimension here - will fail if the two audios were not the same size
        audio_start_end = torch.stack(audio_start_end, dim=0)

        # Code similar to RAVE::validation_step
        z = rave_model.encode(audio_start_end)
        assert isinstance(rave_model.encoder, rave.blocks.VariationalEncoder)
        z_start_end = torch.split(z, z.shape[1] // 2, 1)[0]  # Keep the means only
        # y = rave_model.decode(z_mean)

        # Reparametrization: samples from the posterior (including added gaussian noise).
        # Interesting sound effects ("musical noise"), but definitely worse reconstructions: don't use here
        # z_sampled = rave_model.encoder.reparametrize(z)[0]  # [1] would be the KLD
        # y_from_sampled_z = rave_model.decode(z_sampled)

        # Actually compute the morph here - linear latent interpolation
        alpha = (torch.arange(0.0, n_steps) / (n_steps - 1.0)).view(n_steps, 1, 1)
        z_interp = ((1.0 - alpha) * z_start_end[0:1, :, :]) + (alpha * z_start_end[1:2, :, :])
        audio_interp = rave_model.decode(z_interp)
        # Crop: trim the tail only. The head contains non-null values from the very first samples,
        #    probably thanks to the non-causal convolutions for a non-realtime RAVE.
        audio_interp = audio_interp[:, :, 0:audio_start_end.shape[2]]
        for step_idx in range(audio_interp.shape[0]):
            # Resample to the original sr, then save
            y = librosa.resample(y=audio_interp[step_idx, 0, :].numpy(), orig_sr=rave_model.sr, target_sr=in_out_sr)
            sf.write(seq_dir.joinpath(f'audio_step{step_idx:02d}.wav'), y, in_out_sr)



if __name__ == "__main__":
    app.run(main)
