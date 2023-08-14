output_path=$1


python evals/evals_qafactevals.py \
--predicted_path $output_path \
--gold_path $output_path \