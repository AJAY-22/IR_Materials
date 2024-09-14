import json
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config
import os
from tqdm import tqdm
import wandb
# Class to perform Query Likelihood computation
class QueryLikelihood:
    def __init__(self):
        modelName = 'bert-base-uncased'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
        model = BertModel.from_pretrained(modelName)
        
        # Freeze all layers except the last 3
        for i, layer in enumerate(model.encoder.layer):
            if i < 11:  # Freeze layers 0 to 8
                for param in layer.parameters():
                    param.requires_grad = False
            else:  # Unfreeze layers 9, 10, 11
                for param in layer.parameters():
                    param.requires_grad = True

        # Verify which parameters are frozen and which are trainable
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: requires_grad={param.requires_grad}")

        model = nn.DataParallel(model)
        self.model = model.to(self.device)

    def train(self, queries, epoch, batchSize=32):
        self.model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        batchNum = 0
        totalLoss = 0
        with tqdm(  range(0, len(queries), batchSize), 
                    desc=f'Training Epoch {epoch}', 
                    colour='blue') as trainBar:
            for ix in trainBar:
                # Get all the pairs for ``batchSize`` number of queries
                batchNum += 1
                batch = []
                for jx in range(ix, ix+batchSize):
                    record = queries[jx]
                    query = record['query']
                    # for relDoc in record['relDocContent']:
                    relDoc = record['relDocContent'][0]
                    batch.append((relDoc, query, 1)) # 1 as this is relDoc query pair

                    for notRelDoc in record['notRelDocs'][:8]:
                        batch.append((notRelDoc, query, 0)) # 0 as this is noRelDoc query pair

                # print(f'Total Pairs in batch {batchNum} : {len(batch)}')

                docBatch = [pair[0] for pair in batch]
                queryBatch = [pair[1] for pair in batch]
                trueLabels = torch.tensor([pair[2] for pair in batch]).to(self.device)
                
                # Batch encoding using batch_encode_plus
                encoded_inputs = self.tokenizer.batch_encode_plus(
                    list(zip(docBatch, queryBatch)),      # Pairs of documents and queries
                    add_special_tokens=True,            # Add [CLS] and [SEP] tokens
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**encoded_inputs)
                hidden_states = outputs.last_hidden_state   # Shape: (batch_size, sequence_length, hidden_size)

                linear_projection = torch.nn.Linear(hidden_states.size(-1), self.tokenizer.vocab_size).to(self.device)
                logits = linear_projection(hidden_states) # Shape (batch_size, queryLen , vocab_size)
                probs = F.softmax(logits, dim=-1)

                startIdxs = []
                endIdxs = []
                # Find the index where query starts and ends for each input in the batch
                attentionMask = encoded_inputs['attention_mask']
                inputIds = encoded_inputs['input_ids']
                sep_token_id = self.tokenizer.sep_token_id
                sep_positions = (inputIds == sep_token_id).nonzero(as_tuple=True)

                # Batch and sequence indices of SEP tokens
                batch_indices = sep_positions[0]  # Batch indices
                seq_indices = sep_positions[1]     # Sequence indices where [SEP] tokens are located

                # Reshape to get two SEP positions per sequence (since each sequence has two [SEP] tokens)
                sep_positions_reshaped = seq_indices.view(inputIds.size(0), 2)

                # Start and end indices for queries
                startIdxs = sep_positions_reshaped[:, 0] + 1  # Start of query (right after first SEP)
                endIdxs = sep_positions_reshaped[:, 1] - 1    # End of query (just before second SEP)

                # Initialize an empty list to store the sliced probabilities for each sequence
                queryLogLikelihoods = []

                # Loop over each batch element

                for i in range(probs.size(0)):  # probs.size(0) is the batch size
                    queryTokenIds = inputIds[i, startIdxs[i]:endIdxs[i]]
                
                    # here now we will sum over log probs of all query tokens
                    queryProb = 0
                    for tIx, tokenId in enumerate(queryTokenIds):
                        queryProb += torch.log(probs[i, startIdxs[i]+tIx, tokenId] + 1e-9)
                    queryLogLikelihoods.append(queryProb)

                queryLogLikelihoods = torch.stack(queryLogLikelihoods)

                trainLoss = criterion(torch.sigmoid(-queryLogLikelihoods), trueLabels.type(dtype=torch.float))
                trainLoss.backward()
                totalLoss += trainLoss.item()
                optimizer.step()
                optimizer.zero_grad()
                trainBar.set_postfix(RunningTrainLoss = f'{totalLoss/batchNum:.4f}')
                wandb.log({
                    "Running Train Loss": (totalLoss/batchNum)
                })
        return (totalLoss/batchNum)

# Main execution
if __name__ == "__main__":
    totalEpochs = 5
    batchSize = 2
    # Load HotpotQA training data
    trainDumpFile = os.path.join(config.A1_DATA_DUMP, 'hotpotTrain.json')
    with open(trainDumpFile, 'r') as trainFile:
        trainData = json.load(trainFile)
    
    print(len(trainData))
    # Process data and compute query likelihoods

    wandb.init(project='IR Assignment 1')

    model = QueryLikelihood()
    for epoch in range(totalEpochs):
        trainLoss = model.train(trainData, epoch, batchSize=batchSize)
        wandb.log({
            "Epoch": epoch,
            "TrainLoss": trainLoss
        })
        print(f'Epoch {epoch} loss: {trainLoss}')
          
    wandb.finish()