from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import os, argparse, shutil
from multiprocessing import Pool

from pathlib import Path
import configparser

import ltsp.model



# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./default.ini', help='path to the config file')
args = parser.parse_args()

# Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
config.read(config_path)

# audio configs
sampling_rate = config['audio'].getint('sampling_rate')
hop_length = config['audio'].getint('hop_length')
bins_per_octave = config['audio'].getint('bins_per_octave')
num_octaves = config['audio'].getint('num_octaves')
n_bins = num_octaves * bins_per_octave
n_iter = config['audio'].getint('n_iter')
cqt_bit_depth = config['audio'].get('cqt_bit_depth')

# dataset
dataset = Path(config['dataset'].get('datapath'))
if not dataset.exists():
    raise FileNotFoundError(dataset.resolve())

cqt_dataset_dir = dataset / "LTS_CQT"
if cqt_dataset_dir.exists():
    shutil.rmtree(cqt_dataset_dir)
os.makedirs(cqt_dataset_dir, exist_ok=False)

my_audio_folder = dataset / 'audio'
audio_files = [f for f in my_audio_folder.glob('*.wav')]

print('TOTAL FILES: {}'.format(len(audio_files)))

config_path = cqt_dataset_dir / 'config.ini'
with open(config_path, 'w') as configfile:
    config.write(configfile)


def calculate_cqt(f, verbose=False):
    outfile = cqt_dataset_dir.joinpath(f.stem + '.npy')

    try:
        C = ltsp.model.compute_CQT_from_file(
            f, sampling_rate, hop_length, bins_per_octave, n_bins, cqt_bit_depth, verbose
        )
        if verbose:
            print('writing: {}'.format(outfile))
        # Raw: 2 MB for each 4s audio file... almost triples the original dataset's size
        np.save(outfile, C)

    except Exception as e:
        print(f'Exception raised when processing {f}')
        raise e


if __name__ == '__main__':

    # import multiprocessing.dummy
    # pool = multiprocessing.dummy.Pool(processes=1)

    # Quite fast - don't need to tqdm this
    pool = Pool()
    print("Computing Constant-Q Transforms....")
    pool.map(calculate_cqt, audio_files)
    pool.close()
    pool.join()
