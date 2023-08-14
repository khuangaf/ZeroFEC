from zerofec.zerofec import ZeroFEC
from types import SimpleNamespace
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--dataset')
args = parser.parse_args()


def get_fever_model_args():

    args = {
        'qg_path': 'Salesforce/mixqg-base',
        'qa_model_path': 'khhuang/zerofec-daqa-t5-base',
        'qa_tokenizer_path': 'khhuang/zerofec-daqa-t5-base',
        'entailment_model_path': '/shared/nas/data/m1/khhuang3/info_correct/docnli/docnli-roberta_pubmed_bioasq.pt',
        'entailment_tokenizer_path':'roberta-large',
        'qa2s_tokenizer_path': 'khhuang/zerofec-qa2claim-t5-base',
        'qa2s_model_path': 'khhuang/zerofec-qa2claim-t5-base',
        'use_scispacy': False
    }
    # This way the algo can call args with args.model_name, not args["model_name"].
    # So the core code is still compatible with argparse'd arguments
    args = SimpleNamespace(**args)

    return args

def get_scifact_model_args():

    args = {
        'qg_path': 'Salesforce/mixqg-base',
        'qa_model_path': 'khhuang/zerofec-daqa-t5-base',
        'qa_tokenizer_path': 'khhuang/zerofec-daqa-t5-base',
        'entailment_model_path': '/shared/nas/data/m1/khhuang3/info_correct/docnli/docnli-roberta_pubmed_bioasq.pt',
        'entailment_tokenizer_path':'roberta-large',
        'qa2s_tokenizer_path': 'khhuang/zerofec-qa2claim-t5-base',
        'qa2s_model_path': 'khhuang/zerofec-qa2claim-t5-base',
        'use_scispacy': True
    }
    # This way the algo can call args with args.model_name, not args["model_name"].
    # So the core code is still compatible with argparse'd arguments
    args = SimpleNamespace(**args)

    return args    

if 'fever' in args.input_path:
    print("Loading FEVER model")
    model_args = get_fever_model_args()
elif 'scifact' in args.input_path:
    print("Loading SciFact model")
    model_args = get_scifact_model_args()

zerofec = ZeroFEC(model_args)

with open(args.input_path,'r') as f:
    inputs = [json.loads(l) for l in f.readlines()]

outputs = zerofec.batch_correct(inputs)

with open(args.output_path,'w') as f:
    for output in outputs:
        f.write(json.dumps(output)+'\n')