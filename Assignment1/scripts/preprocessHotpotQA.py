import os
import json
import config
from tqdm import tqdm

def processHotpotQA(train_json, output_json):
    with open(train_json, 'r') as infile:
        data = json.load(infile)
    
    processed_data = []
    
    for entry in tqdm(data):
        question = entry['question']
        qId = entry['_id']
        supporting_docs = {fact[0] for fact in entry['supporting_facts']}
        relDocs = []
        nonRelDocs = []
        for context in entry['context']:
            doc_title = context[0]
            doc_content = ' '.join(context[1])  # Concatenate all sentences in the document
            doc_content = f"{doc_title}: {doc_content}"
            is_relevant = 1 if doc_title in supporting_docs else 0
            if is_relevant:
                relDocs.append(doc_content)
            else:
                nonRelDocs.append(doc_content)
            
        # Create a dict for the processed data
        processed_data.append({
            'qId': qId,
            'query': question,
            'relDocContent': relDocs,
            'notRelDocs': nonRelDocs
        })

    outputFilePath = os.path.join(config.A1_DATA_DUMP, output_json)
    with open(outputFilePath, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)

if __name__ == '__main__':
    trainJson = os.path.join(config.A1_DATA_DUMP, 'hotpot_train_v1.1.json')
    outputFile = os.path.join(config.A1_DATA_DUMP, 'hotpotTrain.json')
    # valJson = os.path.join(config.A1_DATA_DUMP, 'hotpot_dev_distractor_v1.json')

    processHotpotQA(trainJson, outputFile)