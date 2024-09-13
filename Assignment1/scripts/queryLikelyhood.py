import json
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
print(sys.executable)
import config
import os
from tqdm import tqdm
# Class to perform Query Likelihood computation
class QueryLikelihood:
    def __init__(self):
        modelName = 'bert-base-uncased'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
        self.model = BertModel.from_pretrained(modelName).to(self.device)

    def computeLogLikelihood(self, qeuries, batchSize=32):
        pairs = []

        for ix in tqdm(range(0, len(qeuries))):
            record = qeuries[ix]
            query = record['query']
            for relDoc in record['relDocContent']:
                pairs.append((relDoc, query, 1)) # 1 as this is relDoc query pair

            for notRelDoc in record['notRelDocs']:
                pairs.append((notRelDoc, query, 0)) # 0 as this is noRelDoc query pair

        print(f'Total Pairs: {len(pairs)}')

        print(f'Training...')
        for ix in tqdm(range(0, len(pairs), batchSize)):        
            batch = pairs[ix:ix+batchSize]
            docBatch = [pair[0] for pair in batch]
            queryBatch = [pair[1] for pair in batch]

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
            logits = linear_projection(query_hidden_states) # Shape (batch_size, queryLen , vocab_size)
            probs = F.softmax(logits, dim=-1)

            startIdxs = []
            endIdxs = []
            # Find the index where query starts and ends for each input in the batch
            for i, mask in enumerate(encoded_inputs['attention_mask']):
                sep_token_id = self.tokenizer.sep_token_id
                sep_positions = (encoded_inputs['input_ids'] == sep_token_id).nonzero(as_tuple=True)[1]
                startIdxs.append(sep_positions[0].item() + 1)
                endIdxs.append(sep_positions[1].item() - 1)

            # startIdxs and endIdxs are indices where query tokens starts and ends
            # this will be 30k dimensional vocab vector 
            # apply softmax over last dimension (dim=-1) and then find probability for tokens
            

        return log_likelihood.item()

# Main execution
if __name__ == "__main__":
    # Load HotpotQA training data
    trainDumpFile = os.path.join(config.A1_DATA_DUMP, 'hotpotTrain.json')
    with open(trainDumpFile, 'r') as trainFile:
        trainData = json.load(trainFile)
         
    # Process data and compute query likelihoods
    model = QueryLikelihood()
    likelihoods = model.computeLogLikelihood(trainData, batchSize=1)

    # Print out the computed likelihoods for all samples
    print(likelihoods)
