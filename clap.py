import warnings

import torch
from transformers import ClapAudioModel, ClapAudioModelWithProjection, ClapFeatureExtractor, ClapProcessor

from autrainer.models.abstract_model import AbstractModel
from autrainer.models.ffnn import FFNN


class CLAPBackbone(AbstractModel):
    def __init__(
        self,
        model_name,
        freeze_extractor: bool = True,
        time_pooling: bool = True,
    ) -> None:
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        self.time_pooling = time_pooling
        
        model = ClapAudioModelWithProjection.from_pretrained(self.model_name)
        super().__init__(output_dim=model.config.hidden_size)

        self.model = model.audio_model.audio_encoder
        # self.model = model
        # print(self.model)

        if self.freeze_extractor:
            for param in self.model.parameters():
                param.requires_grad = False

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        is_longer = torch.tensor([False])

        x = self.model(input_features=inputs, is_longer=is_longer).last_hidden_state

        # Flatten and transpose for the embeddings
        x = x.flatten(2).transpose(1, 2)

        if self.time_pooling:
            x = x.mean(1)

        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.embeddings(features)


class CLAPFFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        freeze_extractor: bool,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """CLAP model with FFNN frontend adapted for audio classification.
        For more information, see: https://huggingface.co/docs/transformers/en/model_doc/clap#clap

        Args:
            output_dim: Output dimension of the FFNN.
            model_name: Name of the model loaded from Huggingface.
            freeze_extractor: Whether to freeze the feature extractor.
            hidden_size: Hidden size of the FFNN.
            num_layers: Number of layers of the FFNN. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
        """
        super().__init__(output_dim)
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = CLAPBackbone(
            model_name=model_name,
            freeze_extractor=freeze_extractor,
            time_pooling=True,
        )
        self.frontend = FFNN(
            input_size=self.backbone.output_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.frontend(self.embeddings(features)) 


if __name__=='__main__':
    output_dim = 4
    model_name = "laion/clap-htsat-fused"
    freeze_extractor = True
    time_pooling = True
    hidden_size = 512

    model = CLAPFFNN(
        output_dim=output_dim,
        model_name = model_name,
        freeze_extractor = freeze_extractor,
        hidden_size=hidden_size
    )

    feature_extractor = ClapFeatureExtractor.from_pretrained('laion/clap-htsat-unfused')
    # processor = ClapProcessor.from_pretrained('laion/clap-htsat-unfused')

    import librosa
    a, sr = librosa.load("/path/to/example.wav", sr=48000)
    print(a.shape, sr)
    audio = torch.tensor(a)

    # inputs = processor(audios=audio, sampling_rate=48000, return_tensors="pt")
    # print("Inputs:", inputs['input_features'].shape)
    extracted = feature_extractor(audio, sampling_rate=48000, return_tensors='pt')
    print("Extracted: ", extracted['input_features'].shape)
    extracted = extracted['input_features']
    # print(type(inputs))
    print(type(extracted))

    # features = extracted[list(extracted.keys())[0]][0].unsqueeze(0)
    out = model(extracted)
    print(out)
    print(out.shape)