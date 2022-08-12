import json
from nltk.tokenize import word_tokenize
import jsonlines
import pickle
import pandas as pd

"""
f = open('data/NaMnO_both.json')
data = json.load(f)
len(data)

sentences = []
for i in range(len(data)):
    if len(data[i]['Label']['classifications']) > 1:
        if data[i]['Label']['classifications'][1]['answer']['value'] == 'both_e_g_a_sentence_that_relates_micro_to_macro_challenges':
            sentences.append(data[i]['Labeled Data'])
"""

data = pd.read_csv('data/dataset_labelled.csv')
cat_list = []
for i in range(len(data)):
    if data['challenges_type'][i] in ['both_e_g_a_sentence_that_relates_micro_to_macro_challenges']:
        cat_list.append(data['text'][i])

data = cat_list
list_dicts = []
for i in range(len(data)):
    paper_dict = {}
    paper_dict['doc_key'] = "covid-event"+str(i)
    paper_dict['relations'] = []
    paper_dict['ner'] = []
    paper_dict['relations'].append([])
    paper_dict['ner'].append([])
    paper_dict['sentences'] = [word_tokenize(data[i])]
    list_dicts.append(paper_dict)

with jsonlines.open('data/5-6_important_challenges_test.json', mode='w') as writer:
    for elem in list_dicts:
        writer.write(elem)
print('done')


{"doc_key": "yj2kunat_abstract", "sentences": [["However",",", "the", "sluggish", "materials" ,"diffusion", "and", "poor", "electronic", "conductivity", "of", "NVP", "often", "hinder" ,"electrochemical", "performance",",", "thus", "requiring", "compositing", "with", "carbon" ,"materials",",", "such" ,"as" ,"graphene", "to", "improve", "the", "material","."]],  "relations": [[]], "ner": [[]]}

