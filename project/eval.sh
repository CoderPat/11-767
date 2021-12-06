export GLUE_DIR=/data/datasets/GLUE
export TASKS="CoLA MNLI MRPC QNLI QQP RTE SST-2 STS-B WNLI"

export MODEL_NAME=distilbert
export OUTPUT_DIR=/data/output

for TASK in $TASKS; do
  CUDA_VISIBLE_DEVICES=2 python ./run_glue_with_pabee.py \
    --model_type $MODEL_NAME \
    --model_name_or_path /$OUTPUT_DIR/$TASK_NAME/checkpoint-500 \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --data_dir "$GLUE_DIR/$TASK_NAME" \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --logging_steps 50 \
    --num_train_epochs 15 \
    --output_dir ./output/ \
    --eval_all_checkpoints \
    --tokenizer $MODEL_NAME-base-uncased \
    --patience 1,2,3,6 \
    --benchmark
  done