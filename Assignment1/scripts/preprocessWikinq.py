import json

# Input and output file paths
input_file = 'nq-train-new.jsonl'
output_file = 'nq-train-old.jsonl'

# Initialize data structures
queries = {}

# Read and process the JSONL file
with open(input_file, 'r') as infile:
    for line in infile:
        record = json.loads(line)
        query_id = record['query_id']
        query = record['query']
        doc_text = record['text']
        label = record['label']
        
        if query_id not in queries:
            queries[query_id] = {
                'query': query,
                'relDocContent': [],
                'notRelDocs': []
            }
        
        if label == 1:
            queries[query_id]['relDocContent'].append(doc_text)
        else:
            queries[query_id]['notRelDocs'].append(doc_text)

# Write the processed data to a new JSONL file
with open(output_file, 'w', encoding='utf-8') as outfile:
    for query_id, data in queries.items():
        output_record = {
            'qid': query_id,
            'query': data['query'],
            'relDocContent': data['relDocContent'],
            'notRelDocs': data['notRelDocs']
        }
        outfile.write(json.dumps(output_record) + '\n')

print(f'Conversion complete. Data saved to {output_file}')
