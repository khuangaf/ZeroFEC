from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import argparse
import json
from tqdm import tqdm
import spacy 
import stanza
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--use_scispacy', action='store_true')

args = parser.parse_args()


os.makedirs('/'.join(args.output_path.split('/')[:-1]), exist_ok=True)

if args.use_scispacy:
    nlp = spacy.load('en_core_sci_md')
else:
    nlp = spacy.load('en_core_web_lg')

stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')


with open(args.input_path,'r') as f:
    samples = [json.loads(l) for l in f.readlines()]


def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = [] 
    for child in tree.children:
        results += get_phrases(child, label)
    
    
    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results

for sample in tqdm(samples, desc="Candidate Generation"):

    doc = nlp(sample['mutated'])
    stanza_doc = stanza_nlp(sample['mutated'])
    
    ents = [ent.text for sent in doc.sents for ent in sent.noun_chunks] 
    ents += [ent.text  for sent in doc.sents for ent in sent.ents]
    ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
    ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
    ents += [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos in ['VERB','ADV','ADJ','NOUN']]
    


    # negation
    negations = [word for word in ['not','never'] if word in sample['mutated']]

    # look for middle part: relation/ verb
    middle = []
    start_match = ''
    end_match = ''
    for ent in ents:
        # look for longest match string
        if sample['mutated'].startswith(ent) and len(ent) > len(start_match):
            start_match = ent
        if sample['mutated'].endswith(ent+'.') and len(ent) > len(end_match):
            end_match = ent
    
    
    if len(start_match) > 0 and len(end_match) > 0:
        
        middle.append(sample['mutated'][len(start_match):-len(end_match)-1].strip())
        
    sample['candidate_answers'] = list(set(ents + negations + middle))
    


with open(args.output_path ,'w') as f:
    for sample in samples:
        f.write(json.dumps(sample)+'\n')
