import spacy
import pandas as pd
import json
import re


# This function converts spaCy docs to the list of named entity spans in Label Studio compatible JSON format:
def doc_to_spans(doc, token_list):
    results = []
    for entity in doc:
        text = ' '.join(token_list[entity[0]:entity[1]+1])
        sentence = ' '.join(token_list)
        #match = re.search(text, ' '.join(token_list))
        results.append({
            'from_name': 'label',
            'to_name': 'text',
            'type': 'labels',
            'value': {
                'start': sentence.find(text),
                'end': sentence.find(text)+len(text),
                'text': sentence[sentence.find(text):sentence.find(text)+len(text)],
                'labels': ['entity']
            }
        })

    return results


# Now load the dataset and include only lines containing "Easter ":
jsonObj = pd.read_json(path_or_buf='../predictions/5-6_important_challenges_test.jsonl', lines=True)

# Prepare Label Studio tasks in import JSON format with the model predictions:
entities = set()
tasks = []
for k in range(len(jsonObj)):
    predictions = []
    model_pred = jsonObj.iloc[k]['predicted_ner'][0]
    sentence = jsonObj.iloc[k]['sentences'][0]
    spans = doc_to_spans(model_pred, sentence)
    predictions.append({'model_version': 'coarse', 'result': spans})
    text = ' '.join(sentence)
    tasks.append({
        'data': {'text': text},
        'predictions': predictions
    })

# Save Label Studio tasks.json
print(f'Save {len(tasks)} tasks to "d.json"')
with open('../output_label_studio/tasks.json', mode='w') as f:
    json.dump(tasks, f, indent=2)

