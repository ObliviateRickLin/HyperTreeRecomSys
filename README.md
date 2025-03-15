# HyperTreeRecomSys

⚠️ **WORK IN PROGRESS** - This project is under active development and is not yet ready for production use.

## Overview

HyperTreeRecomSys is a novel recommendation system that leverages hyperbolic embeddings to better represent hierarchical product relationships. The system combines the power of hierarchical product taxonomies with modern deep learning techniques to provide more accurate and interpretable product recommendations.

## Project Structure

The project is organized into several components:

- `hierarchical-encoder/`: Module for learning product taxonomy embeddings in hyperbolic space
  - Processes Amazon product hierarchies
  - Trains embeddings using HierarchyTransformers
  - Evaluates embedding quality

## Current Status

- [x] Implemented Amazon Beauty dataset taxonomy construction
- [x] Set up training pipeline for hierarchy learning
- [ ] Add recommendation system integration
- [ ] Implement evaluation metrics
- [ ] Add visualization tools
- [ ] Complete documentation

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HyperTreeRecomSys.git
cd HyperTreeRecomSys
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

To train the hierarchical encoder on the Amazon Beauty dataset:

```bash
cd hierarchical-encoder
python run_train.py --data_dir=datasets/amazon_beauty_dataset/mixed --output_dir=output/amazon-hierarchy
```

## Technical Approach

The system works by:

1. Constructing product taxonomies from Amazon catalog data
2. Learning hyperbolic embeddings of the taxonomy using HierarchyTransformers
3. Utilizing these embeddings to enhance recommendation accuracy

The hyperbolic space is particularly well-suited for representing hierarchical structures, as it can capture both hierarchical relationships and semantic similarity in a compact representation.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built upon [HierarchyTransformers](https://github.com/KRR-Oxford/HierarchyTransformers) for hyperbolic learning
- Uses the Amazon Beauty dataset for taxonomy construction 