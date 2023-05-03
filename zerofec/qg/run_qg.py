from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import json
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base').cuda()

if args.input_path.endswith('.jsonl'):
    with open(args.input_path,'r') as f:
        test_dataset = [json.loads(l) for l in f.readlines()]
else:
    test_dataset = json.load(open(args.input_path,'r'))

def format_inputs(context: str, answer: str):
    return f"answer:{answer} context:{context}"
    # return f"{answer} \\n {context}"

outputs = []

batch_size=10
for test_sample in tqdm(test_dataset, desc="Generating Questions"):
    manipulated_sentence = test_sample['mutated']
    test_sample['generated_question'] = []
    
    for idx in range(0, len(test_sample['candidate_answers']), batch_size):
        texts = [format_inputs(manipulated_sentence, candidate) for candidate in test_sample['candidate_answers'][idx:idx+batch_size]]
        input_ids = tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=1024).input_ids.cuda()
        generated_ids = model.generate(input_ids, max_length=32, num_beams=4)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        test_sample['generated_question'] += output
    # for candidate in test_sample['candidate_answers']:
    #     text = format_inputs(manipulated_sentence, candidate)

    #     input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    #     generated_ids = model.generate(input_ids, max_length=32, num_beams=4)
    #     output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #     test_sample['generated_question'].append(output[0])
    

# with open(args.output_path, 'w') as f:
#     json.dump(test_dataset, f)

with open(args.output_path ,'w') as f:
    for sample in test_dataset:
        f.write(json.dumps(sample)+'\n')