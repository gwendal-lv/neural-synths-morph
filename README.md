
First, create a venv based on python 3.11 and activate it. Then install the following:

```pip3 install numpy pandas scipy effortless_config SoundFile einops librosa tqdm pyyaml```

Instructions from https://pytorch.org/get-started/locally/ (with a specific CUDA 11.8 version):

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

As of May 2024, Tensorflow requires Python >= 3.9 and <= 3.11. Tensorflow dependency: only for CREPE pitch estimation (during preprocessing).
CPU only to prevent conflicts with PyTorch's CUDA packages in the venv.

```pip3 install tensorflow```

CREPE install does something weird with pip: console output stuck, terminal does not respond anymore. It seems to be OK though :

```pip3 install crepe```
