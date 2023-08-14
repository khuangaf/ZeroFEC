

import argparse
import json
import numpy as np
from qafacteval import QAFactEval
import os

parser = argparse.ArgumentParser()
parser.add_argument('--predicted_path', required=True)
parser.add_argument('--gold_path', required=True)
parser.add_argument('--eval_evidence', action='store_true')

args = parser.parse_args()


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

qafacteval_results = metric.score_batch_qafacteval(evidences, predictions, return_qa_pairs=True)
qafacteval_scores = [result[0]['qa-eval']['lerc_quip'] for result in qafacteval_results]
print("QAFactEval: ", np.mean(qafacteval_scores))
