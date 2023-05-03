

from bart_score import BARTScorer
import argparse
import json
import numpy as np
from qafacteval import QAFactEval
import os

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

with open(args.predicted_path,'r') as f:
    predictions = json.load(f)#['predictions']
    predictions = [[pred] for pred in predictions]
tgt_path = args.evidence_path if args.eval_evidence else args.gold_path
with open(tgt_path, 'r') as f:
    tgts = json.load(f)#['target']
    # if not args.eval_evidence:
    # tgts = [[tgt] for tgt in tgts]
with open(args.input_path, 'r') as f:
    inputs = json.load(f)#['source']




kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
            "verbose": True, "generation_batch_size": 8, \
            "answering_batch_size": 8, "lerc_batch_size": 2}

model_folder = 'QAFactEval/models'
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)        
# print(tgts[0])
print(predictions[0])
# print(len(tgts), len(predictions))
# print(set([len(tgt) for tgt in tgts]))
qafacteval_results = metric.score_batch_qafacteval(tgts, predictions, return_qa_pairs=True)
qafacteval_scores = [result[0]['qa-eval']['lerc_quip'] for result in qafacteval_results]
print("QAFactEval: ", np.mean(qafacteval_scores))

predicted_dir = '/'.join(args.predicted_path.split('/')[:-1])
with open(os.path.join(predicted_dir, 'qafact_eval.json'),'w') as f:
    json.dump(qafacteval_results, f)