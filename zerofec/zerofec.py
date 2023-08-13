from .models.answer_selector import AnswerSelector
from .models.question_generator import QuestionGenerator
from .models.question_answerer import QuestionAnswerer
from .models.candidate_generator import CandidateGenerator
from .models.entailment_model import EntailmentModel
from tqdm import tqdm
from typing import List, Dict

import nltk
nltk.download('punkt')

class ZeroFEC:

    def __init__(self, args) -> None:
        # init all the model
        
        
        self.answer_selector = AnswerSelector(args)    
        self.question_generator = QuestionGenerator(args)
        self.question_answerer = QuestionAnswerer(args)
        self.candidate_generator = CandidateGenerator(args)
        self.entailment_model = EntailmentModel(args)
        
        
        print("Finish loading models.")

    
    def correct(self, sample: Dict):
        '''
        sample is Dict containing at least two fields:
            inputs: str, the claim to be corrected.
            evidence: str, the list of reference article to check against.
        '''

        sample = self.answer_selector.select_answers(sample)
        sample = self.question_generator.generate_questions(sample)
        sample = self.question_answerer.generate_answers(sample)
        sample = self.candidate_generator.generate_candidate(sample)
        sample = self.entailment_model.run_entailment_prediction(sample)

        
        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]
        