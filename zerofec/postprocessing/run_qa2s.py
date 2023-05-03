import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--generated_answers_path', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()



samples = [json.loads(l) for l in open(args.generated_answers_path,'r').readlines()]


model = AutoModelForSeq2SeqLM.from_pretrained('qa2claim').cuda()

tokenizer = AutoTokenizer.from_pretrained('t5-base')

def format_inputs(question: str, answer: str):
    return f"{answer} \\n {question}"

for sample in tqdm(samples, desc="Converting QA to statements"):
    generated_questions = sample['generated_question']
    generated_answers = sample['answer']
    mutated_sentence = sample['mutated']
    sample['correction'] = []
    for question, answers in zip(generated_questions, generated_answers):
        
        for answer in answers:
            input_text = format_inputs(question, answer)
            input_ids = tokenizer(input_text, return_tensors="pt", padding='longest', truncation=True, max_length=512).input_ids.cuda()
            
            generated_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
            candidate_corrections = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            sample['correction'] += candidate_corrections

with open(args.output_path ,'w') as f:
    for sample in samples:
        f.write(json.dumps(sample)+'\n')

