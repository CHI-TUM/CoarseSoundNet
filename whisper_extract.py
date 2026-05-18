from transformers import (
    WhisperFeatureExtractor,
)

from transformers.audio_utils import mel_filter_bank

class WhisperFeatureExtractor96k(WhisperFeatureExtractor):
    r"""
    Adapted from: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/feature_extraction_whisper.py
    
    Constructs a Whisper feature extractor for audios with a sampling rate of 96kHz..

    This feature extractor inherits from WhisperFeatureExtractor.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        dither (`float`, *optional*, defaults to 0.0):
            Adds dithering. In other words, adds a small Gaussian noise to each frame.
            E.g. use 0.0001 to add dithering with a normal distribution centered
            around 0.0 with standard deviation 0.0001 (assuming [-1,+1] range of raw_speech).
            The value 0.0 means no dithering.
            Dithering has similar effect as `spectrogram(mel_floor=...)`. It reduces
            the high log_mel_fbank values for signals with hard-zero sections,
            when VAD cutoff is present in the signal.
    """

    def __init__(
        self,
        feature_size=80,
        sampling_rate=96000,
        hop_length=960,
        chunk_length=30,
        n_fft=2400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=False,
        **kwargs,
    ):
        # Default to a window size of 25ms and a hop size of 10ms for the mel filters
        if n_fft is None:
            self.n_fft = int(0.025 * sampling_rate)
        if hop_length is None:
            self.hop_length = int(0.01 * sampling_rate)
        
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=padding_value,
            dither=dither,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.dither = dither
        max_frequency = self.sampling_rate // 2
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=max_frequency,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )   

import librosa
import torch
import numpy as np
import numpy.testing

if __name__ == "__main__":
    torch.manual_seed(3)
    x = torch.rand(1, 96000).numpy()
    fs = 96000
    t = WhisperFeatureExtractor96k(
        feature_size=80,
        sampling_rate=96000,
        hop_length=960,
        chunk_length=30,
        n_fft=400,
    )
    print(t.mel_filters.shape)
    print(t.mel_filters)
    foo = t.mel_filters
    t = WhisperFeatureExtractor96k(
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    print(t.mel_filters.shape)
    print(t.mel_filters)
    bar = t.mel_filters
    np.testing.assert_array_almost_equal(foo, bar)
    exit()
    y = t(
        x,
        sampling_rate=16000,
        padding="max_length",
        return_tensors="pt",
    ).input_features
    print(y.shape)

    z = t(
        librosa.resample(x, orig_sr=96000, target_sr=16000),
        sampling_rate=16000,
        padding="max_length",
        return_tensors="pt",
    ).input_features
    print(z.shape)
    np.testing.assert_array_almost_equal(y, z, 2)