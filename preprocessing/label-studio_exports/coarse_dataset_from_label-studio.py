import json
from nltk.tokenize import word_tokenize
import jsonlines
import pickle
import numpy as np

f = open('preprocessing/label-studio_exports/project-6-at-2022-07-03-22-43-55b97014.json')
data = json.load(f)

def get_ner(datapoint):
    #DyGIE needs the entity mentions in a form where they are specified from start to end token, but label-studi
    # annotations outputs the entities by character position..
    sentence = np.array(word_tokenize(datapoint['data']['text']))
    ner_dict = {}
    ner_list = []
    relation_list = []

    for ann in range(len(datapoint['annotations'][0]['result'])):
        if 'value' in datapoint['annotations'][0]['result'][ann]:
            id = datapoint['annotations'][0]['result'][ann]['id']
            label = datapoint['annotations'][0]['result'][ann]['value']['labels'][0]
            start_idx = datapoint['annotations'][0]['result'][ann]['value']['start']
            end_idx = datapoint['annotations'][0]['result'][ann]['value']['end']
            span = np.array(word_tokenize(datapoint['annotations'][0]['result'][ann]['value']['text']))
            for i in range(len(sentence)):
                if len(' '.join(sentence[:i]))+1 == start_idx:
                    start = i
                if len(' '.join(sentence[:i])) == end_idx:
                    end = i-1
            #verify
            if start_idx == 0:
                start = 0
            if end-start+1 == len(span):
                ner_dict[id] = [start, end, 'ENTITY']
                ner_list.append([start, end, 'ENTITY'])
        else:
            from_id = datapoint['annotations'][0]['result'][ann]['from_id']
            to_id = datapoint['annotations'][0]['result'][ann]['to_id']
            type = "MECHANISM"
            relation_list.append([ner_dict[from_id][0], ner_dict[from_id][1], ner_dict[to_id][0], ner_dict[to_id][1], type])
    return ner_list, relation_list


list_dicts = []
for i in range(len(data)):
    paper_dict = {}
    paper_dict['doc_key'] = "abc" + str(i)
    paper_dict['sentences'] = [word_tokenize(data[i]['data']['text'])]
    ner_list, relation_list = get_ner(data[i])
    paper_dict['relations'] = [relation_list]
    paper_dict['ner'] = [ner_list]
    print(i)
    list_dicts.append(paper_dict)


np.random.seed(21)
indices = np.arange(len(list_dicts))
np.random.shuffle(indices)
list_dicts = np.array(list_dicts)[indices]

train_data = list(list_dicts[:int(0.7*len(data))])
test_data = list(list_dicts[int(0.7*len(data)):])

print(len(train_data)+len(test_data) == len(data))

with jsonlines.open('preprocessing/label-studio_exports/complete_dataset.json', mode='w') as writer:
    for elem in list(list_dicts):
        writer.write(elem)


with jsonlines.open('preprocessing/label-studio_exports/train.json', mode='w') as writer:
    for elem in train_data:
        writer.write(elem)


with jsonlines.open('preprocessing/label-studio_exports/test.json', mode='w') as writer:
    for elem in test_data:
        writer.write(elem)
print('done')