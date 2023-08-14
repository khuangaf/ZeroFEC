

from bart_score import BARTScorer
import argparse
import json
import numpy as np
from sari import SARIsent
from typing import List, Dict
import rouge
from transformers import BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--predicted_path', required=True)
parser.add_argument('--gold_path', required=True)
parser.add_argument('--eval_evidence', action='store_true')



args = parser.parse_args()



def calculate_sari(
    input_lns: List[str], output_lns: List[str], reference_lns: List[str]
) -> Dict:
    a, b, c, d = [], [], [], []
    for input, output, ref in zip(input_lns, output_lns, reference_lns):
        
        a_, b_, c_, d_ = SARIsent(input, output, [ref])

        a.append(a_)
        b.append(b_)
        c.append(c_)
        d.append(d_)

    return {
        "sari_avgkeepscore": np.mean(a),
        "sari_avgdelscore": np.mean(b),
        "sari_avgaddscore": np.mean(c),
        "sari_finalscore": np.mean(d),
    }

def compute_rouge(preds, golds, return_all=False):


    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
                           
    scores = evaluator.get_scores(preds, golds)
    avg_r = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
    # avg_r = scores['rouge-l']['f']

    # print("Recall")
    # print({'r1_recall': f"{scores['rouge-1']['r']*100:.2f}",
    # 'r2_recall': f"{scores['rouge-2']['r']*100:.2f}",
    # 'rL_recall': f"{scores['rouge-l']['r']*100:.2f}",
    # })
    tmp_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

    tmp_scores = tmp_evaluator.get_scores(preds, golds)

    if return_all:
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

        scores = evaluator.get_scores(preds, golds)
        return scores['rouge-l']
    else:
        return {'avg_r':avg_r, 
                'r1': scores['rouge-1']['f'],
                'r2': scores['rouge-2']['f'],
                'rL': scores['rouge-l']['f']}
                
def get_factcc_score(predictions, evidences):
    model = BertForSequenceClassification.from_pretrained('/shared/nas/data/m1/khhuang3/info_correct/qgqa/factCC/factcc-checkpoint').cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    assert len(predictions) == len(evidences)
    res = []
    for prediction, evidence in tqdm(zip(predictions, evidences)):
        
        
        # dynamically determine how much input to use
        encoded_ctx = tokenizer.encode(evidence)[:-1] # remove [SEP]
        encoded_prediction = tokenizer.encode(prediction)[1:] # remove [CLS]

        encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_prediction) ] # - [SEP] - encoded_correction

        # print(tokenizer.decode(encoded_ctx_truncated))

        input_ids = torch.LongTensor(encoded_ctx_truncated + [tokenizer.sep_token_id] + encoded_prediction).cuda().unsqueeze(0)
        token_type_ids = torch.LongTensor([0] * (len(encoded_ctx_truncated)+ 1) + [1] * len(encoded_prediction)).cuda().unsqueeze(0)
        attention_mask = torch.LongTensor([1] * len(input_ids)).cuda().unsqueeze(0)
        
        inputs = {'input_ids':input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask}
        

        with torch.no_grad():
            model.eval()
            outputs = model(**inputs)
            logits = outputs[0]
            probs = torch.nn.Softmax(dim=1)(logits)
            correct_prob = probs[0][0].item()
            res.append(correct_prob)
    
    return res
            

with open(args.predicted_path,'r') as f:

    predictions = []
    inputs = []
    for l in f.readlines():
        sample = json.loads(l)
        inputs.append(sample['input_claim'])
        predictions.append(sample['final_answer'])
     

with open(args.gold_path, 'r') as f:
    gts = []
    evidences = []
    for l in f.readlines():
        sample = json.loads(l)
        gts.append(sample['gt_claim'])
        evidences.append(' '.join(sample['evidence']))
    
    

# if args.eval_evidence:
factcc_scores = get_factcc_score(predictions, evidences)
print("FactCC Score", np.mean(factcc_scores))
# print(predictions)
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scores = bart_scorer.score(evidences, predictions, batch_size=4)

print("BART Score", np.mean(bart_scores))

sari_scores = calculate_sari(inputs, predictions, gts)
print(sari_scores)



rouge_scores = compute_rouge(predictions, gts)
print(rouge_scores)