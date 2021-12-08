export GLUE_DIR=/home/haoming.zhang/11-767/PABEE/glue_data
export TASKS="QNLI"

export MODEL_NAME=bert
export OUTPUT_DIR=/home/haoming.zhang/11-767/PABEE/pabee-bert

for TASK in $TASKS; do
  CUDA_VISIBLE_DEVICES=0 python3 ./run_glue_with_pabee.py \
    --model_type $MODEL_NAME-pabee \
    --model_name_or_path /$OUTPUT_DIR \
    --task_name $TASK \
    --do_layer_eval \
    --do_lower_case \
    --data_dir "$GLUE_DIR/$TASK" \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --logging_steps 50 \
    --num_train_epochs 15 \
    --output_dir ./output/ \
    --eval_all_checkpoints \
    --tokenizer $MODEL_NAME-base-uncased \
    --patience 1,2,3,6 
  done
