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
import argparse

# Class to perform Query Likelihood computation
class QueryLikelihood:
    def __init__(self):
        modelName = 'bert-base-uncased'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
        model = BertModel.from_pretrained(modelName)
        model = nn.DataParallel(model)
        self.model = model.to(self.device)

    def getIdxs(self, task, encoded_inputs):
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
        
        if task == 'queryLikelihood':
            # Start and end indices for queries
            startIdxs = sep_positions_reshaped[:, 0] + 1  # Start of query (right after first SEP)
            endIdxs = sep_positions_reshaped[:, 1] - 1    # End of query (just before second SEP)
        elif task == 'docLikelihood':
            startIdxs = 1 # for document start index is always 1 i.e. after CLS token (CLS has index 0)
            endIdxs = sep_positions_reshaped[:, 0] - 1

        return startIdxs, endIdxs

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

                    # ensure 1:8 ration of good to bad pairs
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

                startIdxs, endIdxs = getIdxs(task, encoded_inputs)
                # Initialize an empty list to store the sliced probabilities for each sequence
                logLikelihoods = []

                # Loop over each batch element
                for i in range(probs.size(0)):  # probs.size(0) is the batch size
                    tokenIds = inputIds[i, startIdxs[i]:endIdxs[i]]
                    # here now we will sum over log probs of all query/doc (depending on task) tokens
                    prob = 0
                    for tIx, tokenId in enumerate(tokenIds):
                        prob += torch.log(probs[i, startIdxs[i]+tIx, tokenId] + 1e-9)
                    logLikelihoods.append(prob)

                logLikelihoods = torch.stack(logLikelihoods)

                trainLoss = criterion(torch.sigmoid(-logLikelihoods), trueLabels.type(dtype=torch.float))
                trainLoss.backward()
                totalLoss += trainLoss.item()
                optimizer.step()
                optimizer.zero_grad()
                trainBar.set_postfix(RunningTrainLoss = f'{totalLoss/batchNum:.4f}')
                wandb.log({
                    "Running Train Loss": (totalLoss/batchNum)
                })
        return (totalLoss/batchNum)
    
    def saveModel(self, saveFileName):
        filePath = os.path.join(config.A1_DATA_DUMP, saveFileName)
        torch.save(self.model.state_dict(), filePath)
        print(f'Model saved at {filePath}')

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use [hotpot, wikinq]')
    parser.add_argument('--task', type=str, required=True, help='Name of the task to perform [queryLikelihood, docLikelihood]')
    
    dataset = parser.dataset
    task = parser.task

    totalEpochs = 5
    batchSize = 2
    
    # Load training data
    if dataset == 'hotpot':
        trainFileName = 'hotpotTrain.json'
        testFileName = 'hotpotTest.json'
    elif dataset == 'wikinq':
        trainFileName = 'wikinqTrain.json'
        testFileName = 'wikinqTest.json'

    trainDumpFile = os.path.join(config.A1_DATA_DUMP, trainFileName)
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

    model.saveModel(f'model_{dataset}_{task}.pth')