CUDA_VISIBLE_DEVICES=0,1
MASTER_PORT=25641
model_name_or_path=/root/share/new_models/microsoft/Phi-3-mini-128k-instruct
model_max_length=2048

deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $MASTER_PORT train.py \
    --deepspeed conf/ds_zero2_no_offload.json \
    --do_train \
    --model_name_or_path  ${model_name_or_path} \
    --output_dir /root/lawer-llm/outputs \
    --data_path /root/lawer-llm/inputs/train_data.json \
    --use_lora false \
    --fp16 true \
    --model_max_length ${model_max_length} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --logging_steps 100


# deepspeed: use fp16 instead of bf16
