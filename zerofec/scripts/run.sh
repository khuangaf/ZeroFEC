output_name=$1
output_dir=output/fever/$output_name

candidate_gen_output_path=$output_dir/generated_candidates.jsonl
qg_output_path=$output_dir/generated_questions.jsonl
qa_output_path=$output_dir/generated_answers.jsonl
claim_output_path=$output_dir/generated_claims.jsonl
factcc_output_path=$output_dir/factcc_output.jsonl
docnli_output_path=$output_dir/docnli_output.jsonl
# python 

python candidate_generation/run_candidate_gen.py --input_path ../data/fever_correct/test_gold_evidence.jsonl --output_path $candidate_gen_output_path 

python qg/run_qg.py --input_path $candidate_gen_output_path   --output_path $qg_output_path

python qa/run_qa_unified.py --generated_question_path $qg_output_path --output_path  $qa_output_path

python postprocessing/run_qa2s.py --generated_answers_path $qa_output_path  --output_path $claim_output_path

python postprocessing/run_docnli.py --generated_claim_path $claim_output_path --output_path $docnli_output_path