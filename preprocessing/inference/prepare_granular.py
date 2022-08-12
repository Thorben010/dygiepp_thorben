import json
from nltk.tokenize import word_tokenize
import jsonlines
import pickle

f = open('project-6-at-2022-06-26-22-47-da1e81ec.json')
data = json.load(f)
len(data)

list_dicts = []
for i in range(len(data)):
    paper_dict = {}
    paper_dict['dataset'] = "covid-event"
    paper_dict['doc_key'] = "covid-event"+str(i)
    paper_dict['events'] = []
    paper_dict['events'].append([])
    paper_dict['sentences'] = [word_tokenize(data[i])]
    list_dicts.append(paper_dict)

with jsonlines.open('data/labelled_dict.jsonl', mode='w') as writer:
    for elem in list_dicts:
        writer.write(elem)
print('done')