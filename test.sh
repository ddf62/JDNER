CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/data/pretrain_model/nezha
export DATA_DIR=$CURRENT_DIR/data
export OUTPUR_DIR=$CURRENT_DIR/data/model_data

python $CURRENT_DIR/code/inference_token.py \
    --model_name_or_path=$BERT_BASE_DIR \
    --output_dir=$OUTPUR_DIR/ \
    --data_dir=$DATA_DIR/

