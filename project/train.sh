export GLUE_DIR=/data/datasets/GLUE
export TASKS="CoLA MNLI MRPC QQP RTE SST-2 STS-B WNLI"

export MODEL_NAME=bert-pabee
export OUTPUT_DIR=/data/jaredfer/odml-bert-pabee-reversed-1

for TASK in $TASKS; do
    CUDA_VISIBLE_DEVICES=2 python ./run_glue_with_pabee.py \
      --model_type $MODEL_NAME \
      --model_name_or_path bert-base-uncased \
      --task_name $TASK \
      --do_train \
      --lazy_max_layers 6 \
      --runtime_input_file /home/haoming.zhang/11-767/project/bert_runtime_lazy_6.txt \
      --do_lower_case \
      --data_dir "$GLUE_DIR/$TASK" \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 64 \
      --per_gpu_eval_batch_size 1 \
      --learning_rate 2e-5 \
      --save_steps 1000 \
      --logging_steps 1000 \
      --num_train_epochs 3 \
      --output_dir $OUTPUT_DIR/$TASK  \
      --tokenizer bert-base-uncased \
      --patience 2,3,4,6,8 \
      --evaluate_during_training;
  done

