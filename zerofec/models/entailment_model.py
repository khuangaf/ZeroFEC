import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from typing import Dict
import numpy as np
import rouge

bert_hidden_dim = 1024
# pretrain_model_dir = 'roberta-large'

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
    def __init__(self, tagset_size, pretrain_model_dir):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size
        
        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None) 
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single

class EntailmentModel:
    def __init__(self, args):
        
        entailment_model_path = args.entailment_model_path
        entailment_tokenizer_path = args.entailment_tokenizer_path
        
        self.model = RobertaForSequenceClassification(num_labels, entailment_tokenizer_path).cuda()#.from_pretrained(pretrain_model_dir).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(entailment_tokenizer_path)
        checkpoint = torch.load(entailment_model_path, map_location=f'cuda:0')
        self.model.load_state_dict(checkpoint)
        self.evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        
    def run_entailment_prediction(self, sample: Dict):
        sample['correction_scores'] = []
        sample['rouge_scores'] = []
        sample['correction'] = sample['correction'] + [sample['input_claim']]#+ sample['g_correction']
        for correction in sample['correction']:
            current_correction_scores = []
            rouge_score = self.evaluator.get_scores([correction], [sample['input_claim']])['rouge-1']['f']
            sample['rouge_scores'].append(rouge_score)
            evidence = sample['evidence'] #if 'full_evidence' in sample else sample['evidence_text']
            
            
            
            for _, ctx in enumerate(evidence):
                
                # if is_wikipedia:
                #     ctx = ' '.join([' '.join(e.split('\t')[1:]) for e in ctx.split('\n')])
                # else:
                
                ctx = ' '.join(ctx.split('\t'))
                ctx = ' '.join(ctx.split('\n'))
                # ctx = ' '.join([e for e in ctx.split('\n')])   
                
                # dynamically determine how much input to use
                encoded_ctx = self.tokenizer.encode(ctx)[:-1] # remove [SEP]
                encoded_correction = self.tokenizer.encode(correction)[1:] # remove [CLS]

                encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_correction) ] # - [SEP] - encoded_correction

                input_ids = torch.LongTensor(encoded_ctx_truncated + [self.tokenizer.sep_token_id] + encoded_correction).cuda().unsqueeze(0)
                
                attention_mask = torch.LongTensor([1] * len(input_ids)).cuda().unsqueeze(0)
                
                
                inputs = {'input_ids':input_ids,
                        'input_mask': attention_mask}
                

                with torch.no_grad():
                    self.model.eval()
                    logits = self.model(**inputs)
                    
                    probs = torch.nn.Softmax(dim=1)(logits)
                    
                    correct_prob = probs[0][0].item()
                    current_correction_scores.append(correct_prob)
            if len(current_correction_scores):
                sample['correction_scores'].append(max(current_correction_scores))

        
        
        if sample['correction_scores']:
            
            argmax = np.argmax(np.array(sample['correction_scores']) + np.array(sample['rouge_scores']))
            
            sample['final_answer'] = sample['correction'][argmax]
        else:
            sample['final_answer'] = sample['input_claim']

        return sample