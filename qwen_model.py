import torch
from transformers import Qwen2AudioEncoderConfig, Qwen2AudioEncoder

from autrainer.models.abstract_model import AbstractModel
from autrainer.models.ffnn import FFNN


class Qwen2FNNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        transfer: str = None,
    ) -> None:
        """Qwen2 audio encoder with FFNN frontend adapted for audio classification.

        Qwen2 audio encoder follows the whisper-large-v3 architecture.

        Args:
            model_name: Name of the model loaded from Huggingface.
            hidden_size: Hidden size of the FFNN.
            output_dim: Output dimension of the FFNN.
            num_layers: Number of layers of the FFNN. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
            transfer: path to pretrained state. Defaults to None.
        """
        super().__init__(output_dim)
        self.encoder = Qwen2AudioEncoder(Qwen2AudioEncoderConfig())
        self.transfer = transfer
        if self.transfer is not None:
            self.encoder.load_state_dict(torch.load(self.transfer))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.frontend = FFNN(
            input_size=self.encoder.config.d_model,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).last_hidden_state.mean(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.frontend(self.embeddings(features))


if __name__ == "__main__":
    print(Qwen2AudioEncoderConfig().d_model)
