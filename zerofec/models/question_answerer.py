import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict
from nltk import word_tokenize


def format_inputs_qa(context: str, question: str):
    # return f"extract answers: <hl> Beyonce further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic, Cadillac Records. <hl> Her performance in the film received praise from critics, and she garnered several nominations for her portrayal of James, including a Satellite Award nomination for Best Supporting Actress, and a NAACP Image Award nomination for Outstanding Supporting Actress."
    return f"{question} \n {context}"

class QuestionAnswerer:
    def __init__(self, args):
        
        qa_model_path = args.qa_model_path
        qa_tokenizer_path = args.qa_tokenizer_path
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(qa_tokenizer_path)

    
    def generate_answers(self, sample: Dict):
        generated_questions = sample['generated_question']
        sample['answer'] = []
        for question in generated_questions:
            
            
            
            question_answers = []
            for _, ctx in enumerate(sample['evidence']):

                words = word_tokenize(ctx)
                passage_size = 400
                for i in range(0, len(words), passage_size):
                    context = ' '.join(words[i:i+passage_size])
                    
                    input_ids = self.tokenizer.encode(f"{question} \n {context}", return_tensors='pt').cuda()
                    
                    with torch.no_grad():
                        
                        outputs = self.model.generate(input_ids, num_beams=4, do_sample=False)
                        predict_answer_tokens_string = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]        
                        question_answers.append(predict_answer_tokens_string.strip())
                
            sample['answer'].append(question_answers)
        
        
        return sample