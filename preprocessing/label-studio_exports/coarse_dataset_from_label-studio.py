import json
from nltk.tokenize import word_tokenize
import jsonlines
import pickle
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
#TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown'])

dataset_name = 'NER_dataset_all'
f = open('labelled_data/'+dataset_name+'.json')
data = json.load(f)

def get_ner(datapoint):
    if datapoint['annotations'][-1]['was_cancelled'] == False:
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
                span = word_tokenize(datapoint['annotations'][0]['result'][ann]['value']['text'])

                start = len(word_tokenize(datapoint['data']['text'][:start_idx]))
                end = len(word_tokenize(datapoint['data']['text'][:end_idx]))-1

                #if len(word_tokenize(datapoint['data']['text'])[start:end]) == len(span):
                ner_dict[id] = [start, end, label]
                ner_list.append([start, end, label])
            else:
                from_id = datapoint['annotations'][0]['result'][ann]['from_id']
                to_id = datapoint['annotations'][0]['result'][ann]['to_id']
                type = "MECHANISM"
                relation_list.append([ner_dict[from_id][0], ner_dict[from_id][1], ner_dict[to_id][0], ner_dict[to_id][1], type])
        return ner_list, relation_list
    return [], []


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

print('Check if all sentences are unique')
print(len(list_dicts))
print(np.unique(np.array([' '.join(x['sentences'][0]) for x in list_dicts])).shape)

final_list = []
good_indices = np.where(np.array([len(x['ner'][0]) for x in list_dicts]) != 0)[0]
for x in good_indices:
    final_list.append(list_dicts[x])

np.random.seed(21)
indices = np.arange(len(final_list))
np.random.shuffle(indices)
list_dicts = np.array(final_list)[indices]

train_data = list(final_list[:int(0.7*len(final_list))])
test_data = list(final_list[int(0.7*len(final_list)):])

print(len(train_data)+len(test_data) == len(final_list))

with jsonlines.open('output/'+dataset_name+'_complete_dataset.json', mode='w') as writer:
    for elem in list(final_list):
        writer.write(elem)


with jsonlines.open('output/'+dataset_name+'_train.json', mode='w') as writer:
    for elem in train_data:
        writer.write(elem)


with jsonlines.open('output/'+dataset_name+'_test.json', mode='w') as writer:
    for elem in test_data:
        writer.write(elem)
print('done')