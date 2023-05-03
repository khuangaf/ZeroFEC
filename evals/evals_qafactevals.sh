output_path=$1


export PYTHONPATH=$(pwd):BARTScore:QAFactEval
python evals/evals_qafactevals.py \
--predicted_path $output_path \
--eval_evidence \
# --eval_scifact