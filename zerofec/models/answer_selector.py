from typing import  Dict
import spacy
import stanza


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

class AnswerSelector:
    def __init__(self, args):
        
        # args used during generation
        self.args = args
        if self.args.use_scispacy:
            self.nlp = spacy.load('en_core_sci_md')
            
        else:
            self.nlp = spacy.load('en_core_web_lg')
        self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    

    def select_answers(self, sample: Dict):
        '''
        This function delete time-related information and store it in `time_removed_tweet`.
        '''
        doc = self.nlp(sample['input_claim'])
        stanza_doc = self.stanza_nlp(sample['input_claim'])
        
        ents = [ent.text for sent in doc.sents for ent in sent.noun_chunks] 
        ents += [ent.text  for sent in doc.sents for ent in sent.ents]
        ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
        ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
        ents += [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos in ['VERB','ADV','ADJ','NOUN']]
        


        # negation
        negations = [word for word in ['not','never'] if word in sample['input_claim']]

        # look for middle part: relation/ verb
        middle = []
        start_match = ''
        end_match = ''
        for ent in ents:
            # look for longest match string
            if sample['input_claim'].startswith(ent) and len(ent) > len(start_match):
                start_match = ent
            if sample['input_claim'].endswith(ent+'.') and len(ent) > len(end_match):
                end_match = ent
        
        
        if len(start_match) > 0 and len(end_match) > 0:
            
            middle.append(sample['input_claim'][len(start_match):-len(end_match)-1].strip())
            
        sample['candidate_answers'] = list(set(ents + negations + middle))

        return sample

