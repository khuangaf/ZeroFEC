import torch
from transformers import BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
import torch
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from nltk import word_tokenize
import numpy as np
import rouge

parser = argparse.ArgumentParser()
parser.add_argument('--generated_claim_path', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large'

label_list = ["entailment", "not_entailment"]#, "contradiction"]
num_labels = len(label_list)

class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None) 
        hidden_states_single = outputs_single[1]

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single


samples = [json.loads(l) for l in open(args.generated_claim_path,'r').readlines()]




model = RobertaForSequenceClassification(num_labels).cuda()
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir)
checkpoint = torch.load('docnli/DocNLI.pretrained.RoBERTA.model.pt', map_location=f'cuda:0')



model.load_state_dict(checkpoint)

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

for sample in tqdm(samples, desc="Running DocNLI"):
    sample['correction_scores'] = []
    sample['rouge_scores'] = []
    sample['correction'] = sample['correction'] + [sample['mutated']]#+ sample['g_correction']
    for correction in sample['correction']:
        current_correction_scores = []
        rouge_score = evaluator.get_scores([correction], [sample['mutated']])['rouge-1']['f']
        sample['rouge_scores'].append(rouge_score)
        evidence = sample['full_evidence'] if 'full_evidence' in sample else sample['evidence_text']
        is_wikipedia = True if 'full_evidence' in sample else False
        if isinstance(evidence[0], list):
            evidence = []
        for ctx_idx, ctx in enumerate(evidence):
            
            if is_wikipedia:
                ctx = ' '.join([' '.join(e.split('\t')[1:]) for e in ctx.split('\n')])
            else:
                ctx = ' '.join([e for e in ctx.split('\n')])   
            
            # dynamically determine how much input to use
            encoded_ctx = tokenizer.encode(ctx)[:-1] # remove [SEP]
            encoded_correction = tokenizer.encode(correction)[1:] # remove [CLS]

            encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_correction) ] # - [SEP] - encoded_correction

            

            input_ids = torch.LongTensor(encoded_ctx_truncated + [tokenizer.sep_token_id] + encoded_correction).cuda().unsqueeze(0)
            
            attention_mask = torch.LongTensor([1] * len(input_ids)).cuda().unsqueeze(0)
            
            
            inputs = {'input_ids':input_ids,
                    
                    'input_mask': attention_mask}
            

            with torch.no_grad():
                model.eval()
                logits = model(**inputs)
                
                probs = torch.nn.Softmax(dim=1)(logits)
                
                correct_prob = probs[0][0].item()
                current_correction_scores.append(correct_prob)
        if len(current_correction_scores):
            sample['correction_scores'].append(max(current_correction_scores))

    
    
    if sample['correction_scores']:
        
        argmax = np.argmax(np.array(sample['correction_scores']) + np.array(sample['rouge_scores']))
        
        sample['final_answer'] = sample['correction'][argmax]
    else:
        sample['final_answer'] = sample['mutated']

with open(args.output_path ,'w') as f:
    for sample in samples:
        f.write(json.dumps(sample)+'\n')

answer_only = [sample['final_answer'] for sample in samples]

answer_only_output_path = args.output_path.split('.')[0] + '_answer_only.json' 
with open(answer_only_output_path ,'w') as f:
    json.dump(answer_only, f)        
