from typing import Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def format_inputs(question: str, answer: str):
    return f"{answer} \\n {question}"

class CandidateGenerator:
    def __init__(self, args):
        self.args = args
        qa2s_tokenizer_path = args.qa2s_tokenizer_path
        qa2s_model_path = args.qa2s_model_path
        self.model = AutoModelForSeq2SeqLM.from_pretrained(qa2s_model_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(qa2s_tokenizer_path)

    def generate_candidate(self, sample: Dict):
        generated_questions = sample['generated_question']
        generated_answers = sample['answer']
        
        sample['correction'] = []
        for question, answers in zip(generated_questions, generated_answers):
            
            for answer in answers:
                input_text = format_inputs(question, answer)
                input_ids = self.tokenizer(input_text, return_tensors="pt", padding='longest', truncation=True, max_length=512).input_ids.cuda()
                
                generated_ids = self.model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
                candidate_corrections = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                sample['correction'] += candidate_corrections
    
        return sample