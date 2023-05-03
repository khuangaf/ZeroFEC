

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
parser.add_argument('--input_path', default='data/fever_correct/evals/test_srcs.json')
parser.add_argument('--gold_path', default='data/fever_correct/evals/test_tgts.json')
parser.add_argument('--evidence_path', default='data/fever_correct/evals/test_evidence.json')
parser.add_argument('--eval_evidence', action='store_true')
parser.add_argument('--eval_scifact', action='store_true')


args = parser.parse_args()
if args.eval_scifact:
    args.evidence_path = args.evidence_path.replace('fever','scifact')
    args.gold_path = args.gold_path.replace('fever','scifact')
    args.input_path = args.input_path.replace('fever','scifact')

factcc_path = args.predicted_path.replace('.json','.factcc.json')
bartscore_path = args.predicted_path.replace('.json','.bartscore.json')
sari_path = args.predicted_path.replace('.json','.sari.json')
sariadd_path = args.predicted_path.replace('.json','.sariadd.json')
sarikeep_path = args.predicted_path.replace('.json','.sarikeep.json')
saridelete_path = args.predicted_path.replace('.json','.saridelete.json')
rouge1_path = args.predicted_path.replace('.json','.rouge1.json')
rouge2_path = args.predicted_path.replace('.json','.rouge2.json')
rougel_path = args.predicted_path.replace('.json','.rougel.json')

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

    with open(sari_path, 'w') as f:
        json.dump(d, f)

    with open(sarikeep_path, 'w') as f:
        json.dump(a, f)

    with open(saridelete_path, 'w') as f:
        json.dump(b, f)

    with open(sariadd_path, 'w') as f:
        json.dump(c, f)
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

    with open(rouge1_path, 'w') as f:
        json.dump([s['f'][0] for s in tmp_scores['rouge-1']], f)

    with open(rouge2_path, 'w') as f:
        json.dump([s['f'][0] for s in tmp_scores['rouge-2']], f)

    with open(rougel_path, 'w') as f:
        json.dump([s['f'][0] for s in tmp_scores['rouge-l']], f)

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
    model = BertForSequenceClassification.from_pretrained('/path/to/factcc-checkpoint').cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    assert len(predictions) == len(evidences)
    res = []
    for prediction, evidence in tqdm(zip(predictions, evidences)):
        
        evidence = '\n'.join([' '.join(e.split('\t')[1:]) for e in evidence.split('\n')])
        # print(ctx)
        # print(ctx)
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
    predictions = json.load(f)#['predictions']

tgt_path = args.evidence_path if args.eval_evidence else args.gold_path
with open(tgt_path, 'r') as f:
    tgts = json.load(f)#['target']
    # if not args.eval_evidence:
    #     tgts = [[tgt] for tgt in tgts]
with open(args.input_path, 'r') as f:
    inputs = json.load(f)#['source']

if args.eval_evidence:
    factcc_scores = get_factcc_score(predictions, tgts)
    print("FactCC Score", np.mean(factcc_scores))
    with open(factcc_path, 'w') as f:
        json.dump(factcc_scores, f)
    # print(predictions)
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_scores = bart_scorer.score(tgts, predictions, batch_size=4)

    print("BART Score", np.mean(bart_scores))

    with open(bartscore_path, 'w') as f:
        json.dump(bart_scores, f)




# bart_score = (np.mean(forward_bart_scores) + np.mean(reverse_bart_scores))/ 2

if not args.eval_evidence:
    sari_scores = calculate_sari(inputs, predictions, tgts)
    print(sari_scores)

    

    rouge_scores = compute_rouge(predictions, tgts)
    print(rouge_scores)