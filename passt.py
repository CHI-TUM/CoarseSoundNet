import warnings
import os
import sys
import contextlib

import torch
from hear21passt.base import get_basic_model, get_model_passt

from autrainer.models.abstract_model import AbstractModel
from autrainer.models.ffnn import FFNN

# Suppress image size mismatch warning
warnings.filterwarnings("ignore", message="Input image size .* doesn't match model .*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.autocast.*is deprecated")



@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class PaSSTFFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        hidden_size: int,
        freeze_extractor: bool = True,
        num_layers: int = 2,
        dropout: float = 0.5,
        transfer: str = None,
    ) -> None:
        """PaSST audio encoder with FFNN frontend adapted for audio classification.
        More information under: https://github.com/kkoutini/PaSST

        Args:
            model_name: Name of the pretrained model loaded from the official github repository.
            hidden_size: Hidden size of the FFNN.
            output_dim: Output dimension of the FFNN.
            num_layers: Number of layers of the FFNN. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
            transfer: path to pretrained state. Defaults to None.
        """
        super().__init__(output_dim)
        self.model_name= model_name
        self.freeze_extractor = freeze_extractor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.transfer = transfer

        model = get_basic_model(mode="logits")
        passt_model = get_model_passt(arch=model_name, n_classes=output_dim)

        # Remove classifier head to get pure encoder
        passt_model.head = torch.nn.Identity()
        passt_model.head_dist = torch.nn.Identity()

        if freeze_extractor:
            for param in passt_model.parameters():
                param.requires_grad = False

        self.encoder = passt_model  # Save encoder
        self.mel = model.mel  
        
        # Classifier head
        self.frontend = FFNN(
            input_size=self.encoder.embed_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        with suppress_stdout():
            # print("Input shape: ", x.shape)
            return self.encoder.forward_features(x)[0]
            # return self.encoder(x)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with suppress_stdout():
            if features.ndim == 3 and features.shape[1] == 1:
                features = features.squeeze(1)  # Convert [B, 1, T] → [B, T]
            
            features = self.mel(features)
            features = features.unsqueeze(1)           # → [B, 1, 128, T'] for PaSST

            emb = self.embeddings(features)

            if isinstance(emb, tuple):
                emb = emb[0]
            return self.frontend(emb)



if __name__=='__main__':
    output_dim = 4
    model_name = "passt_s_swa_p16_128_ap476"
    freeze_extractor = True
    time_pooling = True
    hidden_size = 512

    model = PaSSTFFNN(
        output_dim=output_dim,
        model_name = model_name,
        freeze_extractor = freeze_extractor,
        hidden_size=hidden_size
    )

    # feature_extractor = ClapFeatureExtractor.from_pretrained('laion/clap-htsat-unfused')

    import librosa
    a, sr = librosa.load("/path/to/example.wav", sr=32000, duration=10)
    print(a.shape, sr)
    audio = torch.tensor(a).unsqueeze(0)

    model = get_basic_model(mode="logits")
    model.net = get_model_passt(arch=model_name, n_classes=output_dim)

    features = model.mel(audio)
    print(features)
    print(features.shape)

    out = model(audio)
    print(out)
    print(out.shape)
