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
        relDocs = None
        nonRelDocs = None
        rFlg = nFlg = 0
        for context in entry['context']:
            if rFlg and nFlg: break

            doc_title = context[0]
            doc_content = ' '.join(context[1])  # Concatenate all sentences in the document
            doc_content = f"{doc_title}: {doc_content}"
            is_relevant = 1 if doc_title in supporting_docs else 0
            if is_relevant:
                relDoc = doc_content
                rFlg += 1
            else:
                nonRelDoc = doc_content
                nFlg += 1
            
        # Create a dict for the processed data
        processed_data.append({
            'qId': qId,
            'query': question,
            'relDocContent': relDoc,
            'notRelDoc': nonRelDoc
        })

    outputFilePath = os.path.join(output_json)
    with open(outputFilePath, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    trainJson = os.path.join(config.A1_DATA_DUMP, 'hotpot_train_v1.1.json')
    outputFile = os.path.join('hotpotTrain.json')
    processHotpotQA(trainJson, outputFile)