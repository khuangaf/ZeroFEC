from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import argparse
import json
from tqdm import tqdm
from nltk import word_tokenize
parser = argparse.ArgumentParser()
parser.add_argument('--generated_question_path', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

model_name = 'allenai/unifiedqa-v2-t5-base-1251000'

tokenizer = AutoTokenizer.from_pretrained('t5-base')

model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()



samples = [json.loads(l) for l in open(args.generated_question_path,'r').readlines()]



for sample in tqdm(samples, desc="Generating Answers"):

    generated_questions = sample['generated_question']
    sample['answer'] = []
    for question in generated_questions:
        
        
        answer_predicted = 0
        for ctx_idx, ctx in enumerate(sample['evidence_text']):
            current_answers = []

            words = word_tokenize(ctx)
            passage_size = 400
            for i in range(0, len(words), passage_size):
                context = ' '.join(words[i:i+passage_size])
                input_ids = tokenizer.encode(f"{question} \n {context}", return_tensors='pt').cuda()
                
                with torch.no_grad():
                    
                    outputs = model.generate(input_ids, num_beams=4, do_sample=False)
                    predict_answer_tokens_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    current_answers.append(predict_answer_tokens_string.strip())
                    
        sample['answer'].append(current_answers)
    

with open(args.output_path ,'w') as f:
    for sample in samples:
        f.write(json.dumps(sample)+'\n')