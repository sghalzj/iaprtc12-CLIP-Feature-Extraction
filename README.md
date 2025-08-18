# IAPR TC-12 Feature Extraction and Label Generation Toolkit

This repository provides a complete pipeline for processing the [IAPR TC-12](https://www.imageclef.org/photodata) image dataset, including:
- CLIP-based image and text feature extraction
- Multi-label annotation generation from textual descriptions
- Compatibility with downstream multimodal learning tasks

## 📁 Directory Structure (Expected)

dataset/
└── iaprtc12/
└── unpack/
├── images/
│ └── 00/25.jpg ...
└── annotations_complete_eng/
└── 00/25.eng ...

## 🧩 Scripts Overview

### 1. `encode_image_clip.py`

Extracts image embeddings using [CLIP](https://github.com/openai/CLIP) vision model (`ViT-B/16` by default).

- Input: `iaprtc12/unpack/images/**.jpg`
- Output:  
  - `clip_vitb16_image_embeds_ordered.npz` (ordered by key)  
  - `clip_vitb16_image_embeds_ordered.mat` (for MATLAB compatibility)

**Key Parameters**
```python
MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 64
NORMALIZE = False
FP16 = False

2. textcliptc12.py
Extracts sentence-level text embeddings from the XML annotation files using CLIP text encoder. Two modes are supported:

avg – average sentence embeddings

concat – concatenate first M sentence embeddings

Input: iaprtc12/unpack/annotations_complete_eng/**.eng

Output:

clip_vitb16_text_embeds_sentence_avg.npz or concat5.npz

.mat files for MATLAB users

ENCODE_MODE = "avg" or "concat"
MAX_SENT = 5
FP16 = False

3. labeltc12.py
Generates multi-label one-hot vectors based on words extracted from the <DESCRIPTION> tag.

Preprocessing includes:

Lowercasing

Punctuation and digit removal

Stopword filtering

Keeps only the top-K most frequent tokens (default: 255)

Filters tokens by minimum document frequency

Output:

iaprtc12_vocab255.txt – vocabulary

iaprtc12_labels255.npz – keys, vocab, Y matrix

iaprtc12_labels255.mat – MATLAB version

Key Parameters

TOP_K = 255
MIN_DF = 5
USE_STOPWORDS = True

Output Files (All aligned by key)
keys: relative image names like 00/25
vocab255: list of top-K visual concepts
Y_multi: multi-hot label matrix [N × K]
feats: image or text features from CLIP

Dependencies
pip install numpy scipy tqdm
pip install git+https://github.com/openai/CLIP.git
Also requires: torch, Pillow for image loading, matplotlib (optional).

Citation
If you use this codebase, please cite the IAPR TC-12 benchmark and CLIP.

Example Use Cases
Multimodal representation learning
Visual concept tagging
Image-text retrieval
Supervised or weakly-supervised fine-tuning
Cross-Modal hashing retrieval









