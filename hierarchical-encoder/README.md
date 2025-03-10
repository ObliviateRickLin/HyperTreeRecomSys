# Hierarchical Encoder

This project implements a hierarchical encoder for product taxonomy classification based on the HierarchyTransformers framework. It is designed to encode product descriptions in a hierarchical structure that reflects the taxonomy relationships.

## Prerequisites

Before running the training script, please make sure you have the following dependencies installed:

```
torch
sentence-transformers
pandas
numpy
matplotlib
tqdm
pyyaml
geoopt
```

You can install these dependencies using pip:

```
pip install torch sentence-transformers pandas numpy matplotlib tqdm pyyaml geoopt
```

## Dataset

The model is trained on the Amazon Beauty dataset with hard-negative samples. The dataset should be structured as follows:

```
datasets/
  amazon_beauty_dataset/
    mixed/
      train.jsonl
      val.jsonl
      test.jsonl
      entity_lexicon.json
```

Each line in the JSONL files should be a JSON object with the following structure:
```json
{
  "child": "child_node_id",
  "parent": "parent_node_id",
  "hard_negatives": ["negative_node_id1", "negative_node_id2", ...],
  "random_negatives": ["random_node_id1", "random_node_id2", ...]
}
```

The `entity_lexicon.json` file should contain descriptions for each node in the taxonomy.

## Training

To train the model, run the following command:

```
python run_train.py --data_dir=datasets/amazon_beauty_dataset/mixed --output_dir=output/amazon-hierarchy
```

You can customize the training with the following arguments:

- `--data_dir`: Directory containing the dataset files
- `--model_name`: Pre-trained model name or path (default: "sentence-transformers/all-MiniLM-L12-v2")
- `--output_dir`: Directory to save the model and results
- `--num_epochs`: Number of training epochs (default: 20)
- `--train_batch_size`: Training batch size (default: 64)
- `--eval_batch_size`: Evaluation batch size (default: 128)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--seed`: Random seed (default: 42)
- `--clustering_loss_weight`: Weight for clustering loss (default: 1.0)
- `--clustering_loss_margin`: Margin for clustering loss (default: 3.0)
- `--centripetal_loss_weight`: Weight for centripetal loss (default: 1.0)
- `--centripetal_loss_margin`: Margin for centripetal loss (default: 0.5)

## Model

The model is based on HierarchyTransformer from the HierarchyTransformers framework, which extends SentenceTransformer models with hyperbolic geometry capabilities for better representing hierarchical relationships. 

The training uses two main components for the loss function:
1. HyperbolicClusteringLoss: Ensures that child entities are close to their parent entities
2. HyperbolicCentripetalLoss: Ensures hierarchical properties by encouraging child entities to be "inside" their parent entities in the hyperbolic space

## Evaluation

The model is evaluated using standard classification metrics (precision, recall, F1) on the validation and test sets after each epoch. The best model based on F1 score is saved during training.

```
@inproceedings{NEURIPS2024_1a970a3e,
 author = {He, Yuan and Yuan, Moy and Chen, Jiaoyan and Horrocks, Ian},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Language Models as Hierarchy Encoders},
 year = {2024}
}
``` 