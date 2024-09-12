import os
import json
import config

def processHotpotQA(train_json, output_json):
    with open(train_json, 'r') as infile:
        data = json.load(infile)
    
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
    
    outputFilePath = os.path.join(config.A1_DATA_DUMP, output_json)
    with open(outputFilePath, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)

if __name__ == '__main__':
    trainJson = os.path.join(config.A1_DATA_DUMP, 'hotpot_train_v1.1.json')
    outputFile = os.path.join(config.A1_DATA_DUMP, 'hotpotTrain.json')
    # valJson = os.path.join(config.A1_DATA_DUMP, 'hotpot_dev_distractor_v1.json')

    processHotpotQA(trainJson, outputFile)