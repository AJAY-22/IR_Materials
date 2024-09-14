import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import torch.nn.functional as F
import random

# Dataset Class
class DocumentDatasetRandom(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['label'] == 1:
                    self.data.append(record)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        text = item['text']
        label = item['label']
        return query, text, label

class DocumentDatasetInbatch(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.data.append(record)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        text = item['text']
        label = item['label']
        return query, text, label

# Corpus Class
class CorpusDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.data.append(record['text'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Model Definition
class BertContrastiveModel(nn.Module):
    def __init__(self, model_name):
        super(BertContrastiveModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.ffnn = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, encoded_inputs):
        # Encode the inputs
        outputs = self.bert(
            encoded_inputs 
        )
        outcls = outputs.last_hidden_state[:, 0, :]

        # Compute similarity score
        scores = self.ffnn(combined)
        return scores

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        pos_loss = torch.mean(F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)))
        neg_loss = torch.mean(F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))
        return pos_loss + neg_loss

# Training Function
def train(model, dataloader, corpus_dataset, optimizer, device):
    model.train()
    total_loss = 0
    corpus_texts = [text for text in corpus_dataset]
    corpus_size = len(corpus_texts)
    
    for queries, texts, labels in dataloader:
        encoded_inputs = tokenizer(
                    list(zip(docBatch, queryBatch)),      # Pairs of documents and queries
                    add_special_tokens=True,            # Add [CLS] and [SEP] tokens
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                    ).to(device)

        labels = torch.tensor(labels).to(device).unsqueeze(1).float()  # Add extra dimension and convert to float

        optimizer.zero_grad()

        # Compute scores for positive samples
        pos_scores = model(encoded_inputs)
        
        neg_scores = []
        if negativeScheme == 'random':
            # Randomly Sampling 8 negative docs
            for i in range(len(queries)):
                neg_indices = random.sample(range(corpus_size), 8)  # Randomly sample 8 negative indices
                neg_samples = [corpus_texts[idx] for idx in neg_indices]
                
                for negSample in neg_samples:
                    neg_text = tokenizer(negSample, padding=True, truncation=True, return_tensors="pt").to(device)
                    neg_scores.extend(model(queries[i], neg_text).detach().cpu())
            
        elif negativeScheme == 'inBatch':
            neg_scores = []
            for lIx, label in range(enumerate(labels)):
                if label == 0:
                    neg_text = tokenizer(texts[lIx], padding=True, truncation=True, return_tensors="pt").to(device)
                    neg_scores.extend(model(queries[lIx], neg_text).detach().cpu()) 
        
        neg_scores = torch.stack(neg_scores).to(device)
        
        # Compute contrastive loss
        loss_fn = ContrastiveLoss()
        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use [hotpot, wikinq]')
    parser.add_argument('--task', type=str, required=True, help='Name of the task to perform [queryLikelihood, docLikelihood]')
    parser.add_argument('--ns', type=str, required=True, help='Scheme for negative samples [inBatch, random]')
    
    dataset = parser.dataset
    task = parser.task
    negativeScheme = parser.ns

    # Hyperparameters
    model_name = 'bert-base-uncased'
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3

    if dataset == 'hotpot':
        trainFileName = 'hotpotTrain.json'
        testFileName = 'hotpotTest.json'
    elif dataset == 'wikinq':
        trainFileName = 'wikinqTrain.json'
        testFileName = 'wikinqTest.json'

    # Initialize components
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # dataset = DocumentDataset('data.jsonl', tokenizer)
    
    if negativeScheme == 'random':
        dataset = DocumentDatasetRandom(trainFileName, tokenizer)
    elif negativeScheme == 'inBatch':
        dataset = DocumentDatasetInbatch(trainFileName, tokenizer)
    
    corpus_dataset = CorpusDataset('corpus.jsonl', tokenizer)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertContrastiveModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        loss = train(model, dataloader, corpus_dataset, optimizer, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')

    # Save the model
    model.save_pretrained('fine_tuned_bert')
    tokenizer.save_pretrained('fine_tuned_bert')
