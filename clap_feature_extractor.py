from typing import TYPE_CHECKING, List, Optional, Union
import warnings

from audtorch import transforms as AT
from matplotlib.pyplot import get_cmap
import numpy as np
from omegaconf import OmegaConf

import torch
from torchaudio import transforms as TT
import torchlibrosa
from torchvision import transforms as T
from transformers import (
    ASTFeatureExtractor,
    AutoFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
    ClapFeatureExtractor
)

from autrainer.transforms.abstract_transform import AbstractTransform
from autrainer.transforms.smart_compose import SmartCompose
from autrainer.transforms.utils import _to_numpy, _to_tensor



FE_MAPPINGS = {
    "AST": {"fe": ASTFeatureExtractor, "padding": "max_length"},
    "Whisper": {"fe": WhisperFeatureExtractor, "padding": "max_length"},
    "W2V2": {"fe": Wav2Vec2FeatureExtractor, "padding": "longest"},
    "CLAP": {"fe": ClapFeatureExtractor, "padding": "repeatpad"},
    None: {"fe": AutoFeatureExtractor, "padding": "max_length"},
}


class FeatureExtractor(AbstractTransform):
    def __init__(
        self,
        fe_type: Optional[str] = None,
        fe_transfer: Optional[str] = None,
        sampling_rate: int = 16000,
        hop_length: int = 480,
        fft_window_size: int = 1024,
        frequency_min: int = 0,
        frequency_max: int = 14000,
        order: int = -80,
    ) -> None:
        """Extract features from an audio signal using a feature extractor
        from the Hugging Face Transformers library.

        Args:
            fe_type: The class of feature extractor to use in ["AST", "Whisper",
                "W2V2", None]. If None, the AutoFeatureExtractor will be used.
                Defaults to None.
            fe_transfer: The name of a pretrained feature extractor to use.
                If None, the feature extractor will be initialized with default
                values. Defaults to None.
            sampling_rate: The sampling rate of the audio signal. Defaults to
                16000.
            order: The order of the transform in the pipeline. Defaults to -80.

        Raises:
            ValueError: If neither 'fe_type' nor 'fe_transfer' is provided.
        """
        super().__init__(order=order)
        if fe_type is None and fe_transfer is None:
            raise ValueError(
                "Either 'fe_type' or 'fe_transfer' must be provided."
            )
        self.fe_type = fe_type
        self.fe_transfer = fe_transfer
        self.sampling_rate = sampling_rate
        fe_class = FE_MAPPINGS[self.fe_type]["fe"]
        padding = FE_MAPPINGS[self.fe_type]["padding"]

        if self.fe_transfer is not None:
            feature_extractor = fe_class.from_pretrained(self.fe_transfer)
        else:
            feature_extractor = fe_class()
            extractor_dict = {
                k: repr(v) for k, v in feature_extractor.__dict__.items()
            }
            warnings.warn(
                f"{fe_class.__name__} "
                "initialized with default values:\n"
                f"{OmegaConf.to_yaml(extractor_dict)}"
            )

        def extract_features(signal: np.ndarray) -> torch.Tensor:
            if len(signal.shape) == 2:
                signal = signal.mean(0)
            extracted = feature_extractor(
                signal,
                sampling_rate=self.sampling_rate,
                padding=padding,
                return_tensors="pt",
            )
            return extracted[list(extracted.keys())[0]][0]

        self._extract_features = extract_features

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._extract_features(data.numpy())