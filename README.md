# CLICITA: Continual learning for image captioning through improved image-text alignment

Continual image captioning with mitigation of catastrophic forgetting. This repository implements a multi-loss training strategy that improves object awareness and discriminative visual-language alignment, and achieves strong results on the ContCap continual MS COCO split.

- Continual learning for image captioning with noun-aware training and language-guided contrastive learning (LGCL)
- Adaptive dynamic loss weighting for stable multi-loss optimization
- Reproducible results on the ContCap continual MS COCO split

## Overview

Our model beats previous state-of-the-art results set by “ContCap: A Scalable Framework for Continual Image Captioning.”

Method summary:
- Base cross-entropy (CE) loss for caption generation
- Noun-based auxiliary loss using noun-centric prompts for stronger object grounding
- Language-Guided Contrastive Loss (LGCL) across tasks for discriminative alignment in embedding space
- Dynamic weighting of loss terms to avoid domination by any single term

Dynamic weighting:
- Let L_total = L_ce + L_lgcl + L_nouns and β = L_lgcl / L_total
- Final objective: L = L_ce + β · L_lgcl + L_nouns

LGCL details:
- Text embeddings: noun-prompt embeddings from a pretrained text encoder
- Positive pair: image embedding ↔ correct prompt embedding
- Negatives: noun-prompt embeddings from previous tasks
- Margin-based contrastive (e.g., triplet) objective

## Method Diagram

![diagram](https://github.com/user-attachments/assets/fd6e5fe4-2643-406f-b8cb-e0a9aa3605d0)

## Results (TODO: check!!)

Best results on ContCap continual MS COCO split:
| Metric   | Score |
|----------|-------|
| BLEU-1   | 66.4  |
| BLEU-2   | 47.9  |
| BLEU-3   | 31.8  |
| BLEU-4   | 21.8  |
| ROUGE-1  | 41.8  |
| ROUGE-2  | 15.1  |
| ROUGE-L  | 37.0  |
| METEOR   | 26.3  |
| CIDEr    | 64.1  |

All ContCap split results:
| MODEL                                                                                     | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | CIDEr | LGCL              | Time             |
|-------------------------------------------------------------------------------------------|--------|--------|--------|--------|---------|---------|---------|--------|-------|-------------------|------------------|
| NO LGCL                                                                                   | 62.83  | 43.31  | 28.49  | 18.94  | 39.28   | 12.73   | 32.99   | 25.32  | 58.46 | use_lgcl: False   | 31/03/2025 09:45 |
| SUM+DYNAMIC (best)                                                                        | 66.37  | 47.88  | 31.78  | 21.78  | 41.76   | 15.06   | 36.96   | 26.30  | 64.13 | use_lgcl: True    | 02/04/2025 07:25 |
| DYNAMIC                                                                                   | 61.53  | 41.25  | 26.90  | 17.79  | 38.61   | 12.00   | 32.73   | 27.77  | 57.15 | use_lgcl: True    | 05/04/2025 07:52 |
| SUM                                                                                       | 62.02  | 42.35  | 27.85  | 18.55  | 39.18   | 12.62   | 32.97   | 28.32  | 57.02 | use_lgcl: True    | 06/04/2025 06:47 |

Our dataset split results:
| MODEL                                 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | CIDEr | LGCL            | Time             |
|---------------------------------------|--------|--------|--------|--------|---------|---------|---------|--------|-------|-----------------|------------------|
| No LGCL - Our dataset                 | 61.62  | 42.34  | 27.30  | 17.93  | 37.82   | 11.89   | 32.21   | 27.91  | 56.49 | use_lgcl: False | 05/04/2025 09:44 |
| LGCL - SUM+DYNAMIC - Our dataset      | 61.00  | 41.97  | 27.38  | 18.00  | 38.24   | 12.37   | 33.29   | 27.73  | 56.89 | use_lgcl: True  | 06/04/2025 12:55 |

## Installation (TODO: update!)

Requirements:
absl-py==2.1.0
albucore==0.0.14
albumentations==1.4.14
annotated-types==0.7.0
blis==0.7.11
catalogue==2.0.10
certifi==2024.2.2
charset-normalizer==3.3.2
click==7.1.2
clip==1.0
cloudpathlib==0.19.0
colorama==0.4.6
confection==0.1.5
cycler==0.12.1
cymem==2.0.8
einops==0.8.0
et-xmlfile==1.1.0
eval_type_backport==0.2.0
filelock==3.13.4
fonttools==4.51.0
fsspec==2024.3.1
ftfy==6.2.0
h5py==3.11.0
huggingface-hub==0.24.5
idna==3.7
imageio==2.34.1
Jinja2==3.1.3
joblib==1.4.0
kiwisolver==1.4.5
langcodes==3.4.0
language_data==1.2.0
lazy_loader==0.4
marisa-trie==1.2.0
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.5.3
mdurl==0.1.2
mpmath==1.3.0
murmurhash==1.0.10
networkx==3.3
nltk==3.8.1
numpy==1.26.4
opencv-python-headless==4.10.0.84
openpyxl==3.1.5
packaging==24.0
pandas==2.2.2
Pillow==9.2.0
preshed==3.0.9
pycocoevalcap==1.2
pycocotools==2.0.7
pydantic==2.8.2
pydantic_core==2.20.1
Pygments==2.18.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
regex==2024.4.16
requests==2.31.0
rfc3987==1.3.8
rich==13.8.1
rouge-score==0.1.2
safetensors==0.4.3
scikit-image==0.24.0
scikit-learn==1.5.0
scipy==1.13.0
shellingham==1.5.4
six==1.16.0
sklearn==0.0
smart-open==7.0.4
spacy==3.7.6
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.4.8
sympy==1.12
thinc==8.2.5
threadpoolctl==3.5.0
tifffile==2024.8.30
timm==0.6.7
tokenizer==3.4.4
tokenizers==0.19.1
torch==2.2.2+cu121
torchaudio==2.2.2+cu121
torchprofile==0.0.4
torchtext==0.18.0
torchvision==0.17.2
tqdm==4.66.5
transformers==4.40.0
typer==0.12.5
typing_extensions==4.11.0
tzdata==2024.1
urllib3==2.2.1
wasabi==1.1.3
wcwidth==0.2.13
weasel==0.4.1
wrapt==1.16.0

Setup:
```bash
# clone
git clone https://github.com/<your-org>/Taetz_Bordelius_Continual_ImageCaptioning.git
cd Taetz_Bordelius_Continual_ImageCaptioning

# install deps
pip install -r requirements.txt

# optional: noun extraction via spaCy
pip install spacy
python -m spacy download en_core_web_sm
```

## Data Preparation (TODO: update!!!)

ContCap continual MS COCO split:
- Download MS COCO images and annotations (2017 recommended):
  - images: train2017, val2017
  - annotations: annotations/captions_train2017.json, captions_val2017.json
- Obtain the continual split used by ContCap:
  - Use their official split generation scripts or
  - Download precomputed task splits (image ID lists)

Example layout:
```
data/
  coco/
    images/
      train2017/
      val2017/
    annotations/
      captions_train2017.json
      captions_val2017.json
    splits/
      contcap_task_0.json
      contcap_task_1.json
      ...
```

Configure dataset paths either in YAML configs under configs/ or via CLI flags.

## Training (TODO: update!!!)

Example commands (adjust to your entrypoints/configs):
```bash
# Baseline CE + noun loss only
python train.py --config configs/contcap_base.yaml

# Enable LGCL + dynamic weighting
python train.py --config configs/contcap_lgcl.yaml use_lgcl=true dynamic_weighting=true

# Multi-GPU (DDP) example
torchrun --nproc_per_node=4 train.py --config configs/contcap_lgcl.yaml
```

Common flags:
- data.root=path/to/data/coco
- training.seed=42
- optim.lr=5e-5
- optim.weight_decay=0.01
- schedule.max_epochs=10
- lgcl.margin=0.2
- lgcl.memory_size=K
- nouns.extractor=spacy|nltk
- nouns.max_nouns=8
- nouns.template="An image of {nouns}."

## Evaluation and Inference (TODO: update!)

Evaluate on a task:
```bash
python eval.py --config configs/contcap_lgcl.yaml --checkpoint path/to/ckpt.pt
```

Continual evaluation across tasks:
```bash
python eval_continual.py --config configs/contcap_lgcl.yaml --checkpoint path/to/ckpt.pt
```

Caption custom images:
```bash
python caption.py --image path/to/image.jpg --checkpoint path/to/ckpt.pt --num_beams 5
```

## Checkpoints

- Provide links or scripts to download:
  - No-LGCL baseline
  - SUM+DYNAMIC best model
- Include checksum (SHA256) and exact configs used for each.

## Project Structure (TODO: update!)

```
.
├── configs/                 # YAML configs for datasets/models/training
├── src/                     # model, datasets, losses, trainers
├── scripts/                 # data prep, eval, export utilities
├── notebooks/               # analysis and visualization
├── results/                 # logs, metrics, sample captions
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Reproducibility (TODO: update!)

- Set seeds for Python, NumPy, and PyTorch
- Optionally enable torch.use_deterministic_algorithms(True)
- Log library versions, CUDA/driver info, commit hash, and full config
- Save cfgs and checkpoints per run under results/

## Citation (TODO: update later!)

If you use this work, please cite:

BibTeX (update with final metadata/links):
```bibtex
@misc{taetz_bordelius_continual_captioning_2025,
  title        = {Continual Image Captioning with Noun-Aware and Language-Guided Contrastive Learning},
  author       = {Taetz, M. and Bordelius, O. and Contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-org>/Taetz_Bordelius_Continual_ImageCaptioning}},
  note         = {Code and results for continual image captioning}
}
```

## Acknowledgements

We gratefully acknowledge the authors and maintainers of the following projects, which inspired and informed this work:

- ContCap: A Scalable Framework for Continual Image Captioning — https://github.com/giangnguyen2412/Contcap
- Introducing Language Guidance in Prompt Based Continual Learning (LGCL) — https://github.com/gulzainali98/LGCL
- RATT: Recurrent Attention to Transient Tasks for Continual Image Captioning — https://github.com/delchiaro/RATT

## License

This project is licensed under the Apache License, Version 2.0.

- Full text: see the LICENSE file at the repository root
- Online copy: http://www.apache.org/licenses/LICENSE-2.0
