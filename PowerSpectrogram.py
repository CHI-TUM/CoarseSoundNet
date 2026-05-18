from typing import List, Optional, Union
import numpy as np
import torch
import torchlibrosa
from autrainer.transforms import AbstractTransform
from autrainer.datasets.utils import AudioFileHandler


class PowerSpectrogram(AbstractTransform):
    """
    Create a power spectrogram from an audio signal.

    Args:
        window_size: The size of the window.
        hop_size: Hop length.
        sample_rate: The sample rate of the audio signal.
        fmin: The minimum frequency.
        fmax: The maximum frequency.
        ref: The reference amplitude.
        amin: The minimum amplitude.
        top_db: The top decibel.
        power: The power to apply to the magnitude of the spectrogram.
        order: The order of the transform in the pipeline. Defaults to -90.
    """
    def __init__(
        self,
        window_size: int,
        hop_size: int,
        sample_rate: int,
        fmin: int,
        fmax: int,
        ref: float,
        amin: float,
        power: float = 2.0,
        order: int = -90,
    ):
        super().__init__(order=order)
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.ref = ref
        self.amin = amin
        self.power=power

        self._spectrogram = torchlibrosa.stft.Spectrogram(
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            power=self.power
        )
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._spectrogram(data).squeeze(1)

if __name__=='__main__':
    audioloader = AudioFileHandler(target_sample_rate=96000)
    fp="/path/to/example.wav"
    audio = audioloader.load(file=fp)
    print("Audio shape: ", audio.shape)

    sp = "/path/to/examples.npy"
    log_mel_spec = np.load(sp)
    print("Log-Mel Spectrogram shape: ", log_mel_spec.shape)

    power_spectrogram = PowerSpectrogram(
        4096,
        1024,
        96000,
        50,
        48000,
        1.0,
        1e-10
    )
    power_spec = power_spectrogram(audio)
    print("Power Spectrogram shape: ", power_spec.shape)
