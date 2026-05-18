import os
from typing import Optional, Tuple
import torch
import numpy as np
import random
from glob import glob
from autrainer.augmentations import AbstractAugmentation
from autrainer.transforms import RandomCrop


class AddSpectrogramNoise(AbstractAugmentation):
    def __init__(
        self,
        noise_path: str,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        max_noise_factor: float = 0.5,
        crop_size: int = 376,
        spectrogram_type: str = "96k"
    ):
        super().__init__(order, p, generator_seed)

        self.noise_path = os.path.join(noise_path, "log_mel_" + spectrogram_type)
        # self.noise_path = os.path.join(noise_path, "power_spec_" + spectrogram_type) # TODO: handle the options mel/power spec dynamically!
        self.noise_files = glob(os.path.join(self.noise_path, "**/*.npy"), recursive=True)
        self.max_noise_factor = max_noise_factor
        self.crop_size = crop_size
        self._randomCrop = RandomCrop(self.crop_size)

        # Make randomness deterministic by using the generator seed
        self.random_gen = random.Random(generator_seed)
        self.np_random_gen = np.random.default_rng(generator_seed)


    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        """
        Add the noise to the input tensor.
        """
        noise_spectrogram = self.load_noise_spectrogram()
        combined_spectrogram = self.add_noise(x, noise_spectrogram)
        return combined_spectrogram

    
    def load_noise_spectrogram(self) -> torch.Tensor:
        """
        Load a random noise spectrogram and crop it if necessary.
        """
        noise_fp = self.random_gen.choice(self.noise_files)
        noise_spectrogram = torch.tensor(np.load(noise_fp))
        return noise_spectrogram


    def add_noise(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the input spectrogram x.
        """
        if x.shape != noise.shape:
            # Crop the noise spectrogram with the same RandomCrop as the data is cropped with during Training
            self.crop_size = x.shape[1] # Adapt the crop size of the noise spectrogram dynamically to the input spectrogram
            self._randomCrop = RandomCrop(self.crop_size)
            noise = self._randomCrop(noise)
        # Check again if the shapes do match now
        if x.shape != noise.shape:
            raise ValueError(f"The input and the noise tensor must have the same shape. Currently the have the following shape: x.shape={x.shape} and noise.shape={noise.shape}.")

        noise_factor = self.np_random_gen.uniform(0.1, self.max_noise_factor)
        combined_spectrogram = x + (noise_factor * noise)
        return combined_spectrogram


if __name__=='__main__':
    noise_path = "/path/to/noise_spectrograms"
    asn = AddSpectrogramNoise(noise_path=noise_path, generator_seed=0)

    example_input = torch.tensor(np.load("/path/to/example.npy"))
    rc = RandomCrop(301)
    example_input = rc(example_input)
    print(example_input)

    for i in range(5):
        print("\n\n")
        noisy_output = asn.apply(example_input)
        print(noisy_output)

        