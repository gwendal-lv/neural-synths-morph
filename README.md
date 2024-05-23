
# Acknowledgements

PyTorch code for neural synths has been adapted from the following repositories:
- DDSP: https://github.com/acids-ircam/ddsp_pytorch
- DiffWave: https://github.com/gudgud96/diff-wave-synth

TODO : a waveform synth (not wavenet, too slow...) ;
either RAVE, or maybe one with a wavernn ?

# Setup 

First, create a venv based on python 3.9 and activate it. 
<!-- conda create -n neuralsynths python=3.9 ipython -->

Install PyTorch using the official instructions.
<!-- Specific CUDA version (older than the system's CUDA ?), without requiring conda-forge:
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -->

Then install the following:

TODO remove effortless_config (and tensorboardX) if not used anymore...

```pip3 install numpy pandas scipy effortless_config SoundFile einops librosa tqdm pyyaml nnAudio tensorboardX```

As of May 2024, Tensorflow requires Python >= 3.9 and <= 3.11. Tensorflow dependency: only for CREPE pitch estimation (during preprocessing).
<!-- CPU only to prevent conflicts with PyTorch's CUDA packages (conda installed...) in the venv. -->

```pip3 install tensorflow```

CREPE install may behave weirdly with pip: console output stuck, terminal does not respond anymore. 
The install seems to be OK though :

```pip3 install crepe```
