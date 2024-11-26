#%%
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from model import TestTimeTrainingModel
import numpy as np
from tqdm import tqdm

def load_ffl_model():
    ffl = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return ffl

def prepare_data():
    dataset = load_dataset("mteb/scidocs-reranking", split='test')
    queries = [item['query'] for item in dataset]
    posdoc = [item['positive'][0] for item in dataset]
    negdoc = [item['negative'][0] for item in dataset]
    labels = torch.tensor([1, 0]).repeat(len(queries))

    queries = np.repeat(queries, 2)
    docs = np.array([item for pair in zip(posdoc, negdoc) for item in pair])
    labels = torch.tensor([1 if i % 2 == 0 else 0 for i in range(len(docs))])
    
    return queries, docs, labels

def evaluate(predictions, labels, k_values=[1, 5, 10]):
    precisions, recalls = {}, {}
    mrr = 0.0
    for k in k_values:
        correct_at_k = (predictions[:, :k] == labels.unsqueeze(1)).sum().item()
        precisions[k] = correct_at_k / (len(predictions) * k)
        recalls[k] = correct_at_k / labels.sum().item()
    mrr = (1.0 / (predictions.argmax(dim=1) + 1)).mean().item()
    return precisions, recalls, mrr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = TestTimeTrainingModel(AutoModelForMaskedLM.from_pretrained("bert-base-uncased"), tokenizer).to(device)
    ffl = load_ffl_model().to(device)
    queries, docs, labels = prepare_data()

    predictions_ttt = []
    predictions_vanilla = []

    for q, d in tqdm(zip(queries, docs), desc='Training...', total=len(queries)):
        query_doc = f"[CLS] {q} [SEP] {d} [SEP]"
        inputs = tokenizer(query_doc, return_tensors="pt").to(device)

        # Test-Time Training
        embeddings_ttt = model.test_time_training(inputs)
        score_ttt = ffl(embeddings_ttt).item()
        predictions_ttt.append(score_ttt)

        # Vanilla Inference
        embeddings_vanilla = model.forward(inputs)
        score_vanilla = ffl(embeddings_vanilla).item()
        predictions_vanilla.append(score_vanilla)

    predictions_ttt = torch.tensor(predictions_ttt).reshape(-1, len(docs))
    predictions_vanilla = torch.tensor(predictions_vanilla).reshape(-1, len(docs))

    precisions_ttt, recalls_ttt, mrr_ttt = evaluate(predictions_ttt, labels)
    precisions_vanilla, recalls_vanilla, mrr_vanilla = evaluate(predictions_vanilla, labels)

    print(f"Test-Time Training - Precisions: {precisions_ttt}, Recalls: {recalls_ttt}, MRR: {mrr_ttt}")
    print(f"Vanilla Inference - Precisions: {precisions_vanilla}, Recalls: {recalls_vanilla}, MRR: {mrr_vanilla}")

if __name__ == "__main__":
    main()
