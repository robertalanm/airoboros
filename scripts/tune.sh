# declare -a experts

# ${experts[0]} = "augment"
# ${experts[1]} = "followup0"
# ${experts[2]} = "answer0"
# ${experts[3]} = "followup1"
# ${experts[4]} = "answer1"
# ${experts[5]} = "followup2"
# ${experts[6]} = "answer2"
# ${experts[7]} = "followup3"
# ${experts[8]} = "answer3"

        # --num_train_epochs $EPOCH \

# experts=("arxiv" "book" "c4" "common_crawl" "github" "stackexchange" "wikipedia" "openwebtext")
experts=("empathetic" "lima" "platypus" "wyvern")

export MODEL_SIZE=$1
export BATCH_SIZE=$2
EPOCH=3
export BASE_DIR=/home/paperspace/airoboros
export WANDB_API_KEY=f71f8e9c9ab92fe38b3e592042d30163d3449bbb
export WANDB_PROJECT=sybil-180b-v010

rm -rf $BASE_DIR/experts/$WANDB_PROJECT
rm -rf $BASE_DIR/$WANDB_PROJECT

mkdir -p $BASE_DIR/$WANDB_PROJECT
mkdir -p $BASE_DIR/$WANDB_PROJECT/adapters
mkdir -p $BASE_DIR/$WANDB_PROJECT/$EXPERT/checkpoint-$EPOCH/adapter_model

for EXPERT in "${experts[@]}"; do
    echo "Training $EXPERT"
    export WANDB_NAME=$EXPERT-$EPOCH

    accelerate launch qlora/qlora.py \
        --model_name_or_path garage-bAInd/Platypus2-70B-instruct \
        --output_dir $BASE_DIR/experts/$WANDB_PROJECT/$EXPERT \
        --logging_steps 1 \
        --num_train_epochs $EPOCH \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --data_seed 11422 \
        --evaluation_strategy no \
        --eval_dataset_size 2 \
        --max_new_tokens 1024 \
        --dataloader_num_workers 3 \
        --logging_strategy steps \
        --remove_unused_columns False \
        --do_train \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_modules all \
        --bf16 \
        --bits 4 \
        --double_quant \
        --quant_type nf4 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type constant \
        --dataset sybil/training_data/highest_scoring_$EXPERT.jsonl \
        --dataset_format airoboros \
        --model_max_len 2048 \
        --per_device_train_batch_size $BATCH_SIZE \
        --learning_rate 0.0003 \
        --adam_beta2 0.999 \
        --max_grad_norm 0.3 \
        --lora_dropout 0.1 \
        --weight_decay 0.0 \
        --seed 11422 \
        --report_to wandb \
        --gradient_checkpointing True \
        --ddp_find_unused_parameters False \
        --trust_remote_code True \
        --use_auth_token True

        mkdir $BASE_DIR/$WANDB_PROJECT/adapters/$EXPERT
        cp $BASE_DIR/experts/$WANDB_PROJECT/$EXPERT/checkpoint-$EPOCH/adapter_model/* -r $BASE_DIR/$WANDB_PROJECT/adapters/$EXPERT

done

cp $BASE_DIR/sybil/training_data -r $BASE_DIR/$WANDB_PROJECT/training_data
cp $BASE_DIR/sybil/routing_data -r $BASE_DIR/$WANDB_PROJECT/routing_data
