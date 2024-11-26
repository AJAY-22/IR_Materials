# IR_Materials
All the assignments and projects on Information Retrieval 
## All data should be stored at dataDump folders in the parent folders

## Assignment 3: Graph Link Prediction and Zero-Shot Retrieval

This repository contains the implementation of **Graph Link Prediction** and **Zero-Shot Retrieval** tasks. The assignment is divided into two parts, focusing on distinct yet complementary machine learning challenges.

---

### Part 1: Link Prediction on Graphs

#### Problem Statement:
The goal of this part is to perform **link prediction** using **Graph Neural Networks (GNNs)**. The task is to predict whether an edge exists between two nodes in a graph. The main objectives include:
- Training and evaluating **GCN**, **GAT**, and **GIN** models for link prediction.
- Splitting the graph into **train (60%)**, **validation (20%)**, and **test (20%)** sets.
- Using **cosine similarity** as the scoring function to evaluate node embeddings.
- Incorporating **random LSH** and **neural LSH** for efficient inference and retrieval of top-K potential candidate node pairs.

#### Metrics:
- **Precision@K** (K = 1, 5, 10)
- **MRR** (Mean Reciprocal Rank)

---

### Part 2: Zero-Shot Retrieval

#### Problem Statement:
This part focuses on improving the **zero-shot retrieval** capability of **BERT** on scientific documents. The task involves:
- Using **SciDocs** dataset to evaluate performance on retrieval tasks.
- Experimenting with **Test-Time Training (TTT)** to improve BERT’s inference by performing one step of **masked reconstruction** for each input.
- Comparing the zero-shot performance of the standard pipeline and TTT-enhanced pipeline.

#### Key Features:
1. Fine-tuning a **Feed-Forward Layer (FFL)** on the CLS token of BERT using **HotpotQA**.
2. Experimenting with reconstruction on **query + document** and **document only**.
3. Reporting metrics:
   - **Precision@10**
   - **Recall@10**
   - **MRR**

---

### Folder Structure:
- `LinkPrediction/`:
  - `train.py`: Training loop for GNN-based link prediction.
  - `test.py`: Evaluation and metrics computation.
  - `metrics.py`: Precision, Recall, and MRR computation methods.
  - `LSH.py`: Implementation of random and trainable LSH.
  - `prepareDataset.py`: Downloading and preparing dataset.
  - `model.py`: Definition of differnt GNN models
  - `parser.py`: Argument parser for CLI
- `ZSR/`:
  - `fineTune.py`: Fine-tuning the FFL on HotpotQA for Zero-Shot Retrieval.
  - `model.py`: BERT with Test-Time Training support.
  - `preprocessHotpotQA.py`: Data preprocessing and helper functions.
  - `main.py`: Main loop
---

### How to Run:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train GNN models for link prediction:
   ```bash
   python train.py --dataset CoraFull --gnn_method GCN --num_epoch 50
   ```
3. Test the trained models:
   ```bash
   python test.py --dataset CoraFull --gnn_method GCN --trained_model ./saved_models/model.pth
   ```
4. Fine-tune BERT for zero-shot retrieval:
   ```bash
   python ZSR/fineTune.py
   ```
5. Run Zero shot retrieval:
   ```bash
   python ZSR/main.py
   ```
   

## Assignment1
Files available at `Assignment1/scripts/`

write below command to execute different task (queryLikelihood or docLikelihood) with different datasets (hotpotQA or wikiNQ)

For negative sampling `--ns` argument can be take two values `inBatch` or `random`

```python
python likelihood.py --dataset hotpot --task queryLikelihood
python likelihood.py --dataset wikinq --task queryLikelihood
python likelihood.py --dataset hotpot --task docLikelihood
python likelihood.py --dataset wikinq --task docLikelihood

python clsLikelihood.py --dataset [hotpot/wikinq] --task [queryLikelihood/docLikelihood] --ns [inBatch/random]
```

