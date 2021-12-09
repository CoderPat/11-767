export GLUE_DIR=/data/datasets/GLUE
export TASKS="QNLI CoLA MNLI MRPC QQP RTE SST-2 STS-B WNLI"

export MODEL_NAME=bert
export OUTPUT_DIR=/data/jaredfer/odml-bert-pabee-reversed

for TASK in $TASKS; do
  python3 ./run_glue_with_pabee.py \
    --model_type $MODEL_NAME-pabee \
    --model_name_or_path $OUTPUT_DIR/$TASK \
    --output_dir $OUTPUT_DIR/$TASK \
    --task_name $TASK \
    --do_lower_case \
    --data_dir "$GLUE_DIR/$TASK" \
    --tokenizer $MODEL_NAME-base-uncased \
    --save_splitted_checkpoint $OUTPUT_DIR/$TASK/splitted \
    --benchmark
  done
