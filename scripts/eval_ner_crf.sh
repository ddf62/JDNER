CURRENT_DIR=`pwd`
export CUDA_VISIBLE_DEVICES=0
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="JDNER"
#
python3 run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=${OUTPUR_DIR}/JDNER_output/bert/checkpoint-3825/ \
  --task_name=$TASK_NAME \
  --markup='bio' \
  --do_predict \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --eval_max_seq_length=256 \
  --per_gpu_eval_batch_size=24 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output_eval/ \
  --overwrite_output_dir \
  --seed=42 \
  --overwrite_cache
