CUDA_VISIBLE_DEVICES=0,1
MASTER_PORT=25641

deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $MASTER_PORT train.py \
    --deepspeed conf/ds_zero2_no_offload.json \
    --do_train \
    --model_name_or_path /root/share/new_models/microsoft/Phi-3-mini-128k-instruct \
    --output_dir /root/lawer-llm/outputs \
    --use_lora false \
    --fp16 true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \  
    # --model_max_length 512


# deepspeed: use fp16 instead of bf16
