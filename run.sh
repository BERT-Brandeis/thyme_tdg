BATCH=32
MAX_SEQ_LEN=128
LR=3e-5
W_DECAY=0.3
NUM_EPOCH=3
model_seed=42

model=roberta-large-mnli

data_seed=52
input_dir=input_nli_data/general
tdg_nli_train=$input_dir/general_dmn_tdg_nli_train_1207_seed${data_seed}.json
tdg_nli_dev=$input_dir/general_dmn_tdg_nli_dev_1207_seed${data_seed}.json

#tdg_orig_dev=../temporal_dependency_graphs_crowdsourcing-master/tdg_data/dev.txt
#tdg_orig_test=../temporal_dependency_graphs_crowdsourcing-master/tdg_data/test.txt

tdg_orig_dev=temporal_data/general_tdg_data/dev.txt
tdg_orig_test=temporal_data/general_tdg_data/test.txt

filename_prefix=seed${model_seed}_${data_seed}_output_dir_${NUM_EPOCH}epoch_mnli_batch${BATCH}_seqlen${MAX_SEQ_LEN}_lr${LR}_wghtdcy${W_DECAY}
#output_dir=$output_parent_dir/$filename_prefix
output_dir=../tdg_thyme_parsing/weights_on_e2/general-tdg/seed42_52_output_dir_3epoch_mnli_batch32_seqlen128_lr3e-5_wghtdcy0.3
output_dev_filename=${filename_prefix}_dev.txt
output_test_filename=${filename_prefix}_test.txt

parsed_output_dir=parsed_output
eval_output_dir=eval_output

#python run_glue.py --save_steps 2000 --seed $model_seed --report_to none --save_total_limit 1 \
#  --model_name_or_path $model \
#  --train_file $tdg_nli_train \
#  --validation_file $tdg_nli_dev \
#  --do_train \
#  --do_eval \
#  --max_seq_length $MAX_SEQ_LEN \
#  --per_device_train_batch_size $BATCH \
#  --per_device_eval_batch_size $BATCH \
#  --learning_rate $LR \
#  --weight_decay $W_DECAY \
#  --num_train_epochs $NUM_EPOCH \
#  --output_dir $output_dir --overwrite_output_dir

## Parse test
python parsers/run_parser.py --model_dir $output_dir --output_file $output_test_filename \
--input_file $tdg_orig_test --output_dir $parsed_output_dir