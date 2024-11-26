import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from torch import nn, optim
from tqdm import tqdm
import json
import numpy as np
import wandb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def fine_tune_ffl(ffl, model, tokenizer, device, epochs=3, batch_size=16, lr=1e-4):
    file_path = '/mnt/nas/ajaypathak/IR/IR_Materials/Assignment3/ZSR/hotpotTrain.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    optimizer = optim.Adam(ffl.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ffl.train()
    model.eval()  # We freeze the BERT model for faster training

    for epoch in range(epochs):
        epoch_loss = 0
        for ix in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch + 1}"):
            batch = data[ix:ix+batch_size]
            queries = [record['query'] for record in batch]
            reldocs = [record['relDocContent'][0] for record in batch]
            nonReldocs = [record['notRelDoc'][0] for record in batch]

            # Repeat queries
            queries = np.repeat(queries, 2)
            docs = np.array([item for pair in zip(reldocs, nonReldocs) for item in pair])
            labels = torch.tensor([1 if i % 2 == 0 else 0 for i in range(len(docs))]).to(device)
            # Combine question and context
            inputs = tokenizer(
                [f"[CLS] {q} [SEP] {d} [SEP]" for q, d in zip(queries, docs)],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                cls_embeddings = hidden_states[-1][:, 0, :]

            # Forward pass through FFL
            predictions = ffl(cls_embeddings).squeeze(-1)
            predictions = predictions.float()
            labels = labels.float()
            loss = criterion(predictions, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(data):.4f}")
        
        wandb.log({
            "Epoch": epoch,
            "Train Loss": epoch_loss/len(data)
        })

        save_ffl_model(ffl, f"ffl_hotpotqa_final_{epoch}.pth")

    return ffl

def save_ffl_model(ffl, save_path):
    torch.save(ffl.state_dict(), save_path)
    print(f"FFL model saved to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    ffl = load_ffl_model().to(device)

    wandb.init(project="Zero Shot Retrieval", name="HotPotQA FineTune")
    ffl = fine_tune_ffl(ffl, model, tokenizer, device, epochs=10, batch_size=256, lr=1e-4)
    save_ffl_model(ffl, "ffl_hotpotqa_final.pth")
    wandb.finish()