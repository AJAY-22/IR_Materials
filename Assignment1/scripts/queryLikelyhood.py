import json
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Class for loading HotpotQA dataset
class HotpotQADataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=512):
        self.trainData = self._load_data(json_file)
        self.tokenizer = tokenizer
        self.max_len = max_len


    def _processHotpotQA(train_json, output_json):
        with open(train_json, 'r') as infile:
            data = json.load(infile)['data']
        
        processed_data = []
        
        for entry in data:
            question = entry['question']
            supporting_docs = {fact[0] for fact in entry['supporting_facts']}
            
            for context in entry['context']:
                doc_title = context[0]
                doc_content = ' '.join(context[1])  # Concatenate all sentences in the document
                is_relevant = 1 if doc_title in supporting_docs else 0
                
                # Create a dict for the processed data
                processed_data.append({
                    'query': question,
                    'docContent': f"{doc_title}: {doc_content}",
                    'relevance': is_relevant
                })
        
        with open(output_json, 'w') as outfile:
            json.dump(processed_data, outfile, indent=4)


# Class to perform Query Likelihood computation
class QueryLikelihood:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def compute_log_likelihood(self, document, query):
        inputs = self.tokenizer.encode_plus(
            document, 
            query, 
            add_special_tokens=True, return_tensors='pt', truncation=True, max_length=512
        )
        # Pass through BERT to get logits
        outputs = self.model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, hidden_size)
        query_start_idx = (inputs['input_ids'] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[1].item() + 1
        
        # Extract query logits
        query_logits = logits[0, query_start_idx:, :]
        norm_logits = F.softmax(query_logits, dim=-1)
        log_likelihood = torch.sum(torch.log(norm_logits), dim=0)
        return log_likelihood.item()

# Utility function for preprocessing and computing likelihood over HotpotQA dataset
class HotpotQAProcessor:
    def __init__(self, dataset, model_name='bert-base-uncased'):
        self.dataset = dataset
        self.query_likelihood = QueryLikelihood(model_name)

    def process(self, batch_size=8):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        results = []
        for batch in data_loader:
            for idx in range(len(batch['input_ids'])):
                doc_tokens = batch['input_ids'][idx].tolist()
                doc_text = self.dataset.tokenizer.decode(doc_tokens, skip_special_tokens=True)
                query_text = batch['token_type_ids'][idx].tolist()  # Assuming query tokens follow token_type_ids == 1
                likelihood = self.query_likelihood.compute_log_likelihood(doc_text, query_text)
                results.append(likelihood)
        return results


# Main execution
if __name__ == "__main__":
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Load HotpotQA training data
    hotpot_data = HotpotQADataset('hotpot_train_v1.1.json', tokenizer)

    # Process data and compute query likelihoods
    processor = HotpotQAProcessor(hotpot_data, model_name=model_name)
    likelihoods = processor.process(batch_size=4)

    # Print out the computed likelihoods for all samples
    print(likelihoods)
