# CoarseSoundNet
This repository accompanies our paper **CoarseSoundNet: Building a reliable model for ecological soundscape analysis** [(Gebhard et al., 2026)](https://doi.org/10.1016/j.ecoinf.2026.104101).
It contains the code, configurations, and scripts used to train and evaluate our model. 

For inference and usage instructions, please visit our [Hugging Face model page](https://huggingface.co/HearTheSpecies/CoarseSoundNet).

## Repository overview

- [configs_paper/](configs_paper/): Example configuration files for training models with autrainer.
- [analysis/](analysis/): Scripts for evaluating model performance, analysing predictions, and investigating errors.
- [mixing/](mixing/): Scripts used to create the PublicMix data.
- [processing/](processing/): Scripts for creating dataset splits, encoding labels, and preprocessing data.
- [scripts/](scripts/): Helper scripts for downloading, removing, copying, moving, converting, or adjusting files.
- **Root-level Python files:** Custom dataset classes and model implementations that were not available in autrainer or were adapted to our needs.

## Training

The example configurations use [autrainer](https://autrainer.github.io/autrainer/) version **0.6.0**. After installing the project dependencies from [pyproject.toml](pyproject.toml), update the dataset and output paths in the selected configuration files.

For example, run the CLAP training configuration from the repository root:

```bash
autrainer train --config-path "/path/to/configs_paper/clap_all_sets"
```

This runs the parameter combinations defined in `config.yaml`. Change `--config-path` to select another experiment configuration.

## Citation
If you use CoarseSoundNet or this code, please cite our paper. 
```bibtex
@article{Gebhard26-CBA,
  title = {CoarseSoundNet: Building a reliable model for ecological soundscape analysis},
  journal = {Ecological Informatics},
  volume = {99},
  pages = {104101},
  year = {2026},
  issn = {1574-9541},
  doi = {https://doi.org/10.1016/j.ecoinf.2026.104101},
  author = {Alexander Gebhard and Andreas Triantafyllopoulos and Dominik Arend and Sandra Müller and Svenja Schmidt and Michael Scherer-Lorenzen and Björn W. Schuller}
}
```
