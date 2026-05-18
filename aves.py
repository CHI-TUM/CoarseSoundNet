"""
AVES-model from Hagiwara et al. "AVES: Animal Vocalization Encoder based on Self-Supervision"
Using and adapting the provided code from https://colab.research.google.com/drive/1ZmCyxSXtMVde6L_31OUnZRRWHPIxGamh?usp=sharing#scrollTo=CTo6lQgLmgC9
Based on https://github.com/earthspecies/aves
"""

from typing import Optional
import json

import torch
from torchaudio.models import wav2vec2_model

# from .abstract_model import AbstractModel
from autrainer.models.abstract_model import AbstractModel


# Code to use while initially setting up the model
def freeze_embedding_weights(model, trainable):
  """ Freeze weights in AVES embeddings for classification """
  # The convolutional layers should never be trainable
  model.feature_extractor.requires_grad_(False)
  model.feature_extractor.eval()
  # The transformers are optionally trainable
  for param in model.encoder.parameters():
    param.requires_grad = trainable
  if not trainable:
    # We also set layers without params (like dropout) to eval mode, so they do not change
    model.encoder.eval()


# Code to use during training loop, to switch between eval and train mode
def set_eval_aves(model):
  """ Set AVES-based classifier to eval mode. Takes into account whether we are training transformers """
  model.classifier_head.eval()
  model.model.encoder.eval()

def set_train_aves(model):
  """ Set AVES-based classifier to train mode. Takes into account whether we are training transformers """
  # Always train the classifier head
  model.classifier_head.train()
  # Optionally train the transformer of the model
  if model.trainable:
      model.model.encoder.train()


class AvesClassifier(AbstractModel):
    """ Uses AVES Hubert to embed sounds and classify """
    def __init__(
        self, 
        output_dim: int,
        config_path: str, 
        model_path: str, 
        trainable, 
        embedding_dim=768,
        transfer: Optional[str] = None,
    ):
        super().__init__(output_dim)
        self.config_path = config_path
        self.model_path = model_path
        self.trainable = trainable
        self.embedding_dim = embedding_dim
        self.transfer = transfer

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        # Freeze the AVES network
        self.trainable = trainable
        freeze_embedding_weights(self.model, trainable)
        # We will only train the classifier head
        self.classifier_head = torch.nn.Linear(in_features=embedding_dim, out_features=output_dim)
        # self.audio_sr = 16000

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        assert x.ndim == 2, f"Expected 2D input [batch, time], got {x.shape}"
        out = self.model.extract_features(x)[0][-1]

        mean_embedding = out.mean(dim=1)
        return mean_embedding

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(features)
        x = self.classifier_head(x)
        return x


if __name__=='__main__':
    output_dim = 4
    config_path = "/path/to/downloaded_weights/aves-base-all.torchaudio.model_config.json" 
    model_path = "/path/to/downloaded_weights/aves-base-all.torchaudio.pt"
    trainable = False

    model = AvesClassifier(
        output_dim=output_dim,
        config_path=config_path,
        model_path=model_path,
        trainable=trainable
    )

    import librosa
    a, sr = librosa.load("/path/to/example.wav", sr=16000)
    print(a.shape, sr)
    audio = torch.tensor(a).unsqueeze(0)
    print(audio.shape, type(audio))
    out = model(audio)
    print(out)
    print(out.shape)