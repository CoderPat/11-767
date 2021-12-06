export GLUE_DIR=/data/datasets/GLUE
export OUTPUT_DIR=/data/output
export TASKS="CoLA MNLI MRPC QNLI QQP RTE SST-2 STS-B WNLI"

for TASK in $TASKS; do
    CUDA_VISIBLE_DEVICES=2 python ./run_glue_with_pabee.py \
      --model_type $MODEL_NAME \
      --model_name_or_path $MODEL_NAME-base-uncased \
      --task_name $TASK \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir "$GLUE_DIR/$TASK" \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 64 \
      --per_gpu_eval_batch_size 1 \
      --learning_rate 2e-5 \
      --save_steps 500 \
      --logging_steps 1000 \
      --num_train_epochs 1 \
      --output_dir $OUTPUT_DIR/$TASK  \
      --tokenizer $MODEL_NAME-base-uncased \
      --patience 2,3,4,6,8 \
      --evaluate_during_training;
  done

