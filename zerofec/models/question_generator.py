import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict
import numpy as np


def format_inputs(context: str, answer: str):
    # return f"answer:{answer} context:{context}"
    return f"{answer} \\n {context}"
    
class QuestionGenerator:
    def __init__(self, args):
        
        qg_path = args.qg_path
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(qg_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(qg_path)
    

    def generate_questions(self, sample: Dict):
        batch_size = 10
        manipulated_sentence = sample['input_claim']
        sample['generated_question'] = []
        
        for idx in range(0, len(sample['candidate_answers']), batch_size):
            texts = [format_inputs(manipulated_sentence, candidate) for candidate in sample['candidate_answers'][idx:idx+batch_size]]
            input_ids = self.tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=1024).input_ids.cuda()
            generated_ids = self.model.generate(input_ids, max_length=32, num_beams=4)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            sample['generated_question'] += output
        return sample
    
    