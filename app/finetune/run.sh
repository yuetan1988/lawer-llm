deepspeed --include localhost:0,1 train.py \
    --deepspeed conf/ds_zero2_no_offload.json \
    --model_name_or_path /root/share/model_repos/internlm2-base-7b \
    --data_path /root/lawer-llm/inputs \
    --output_dir /root/lawer-llm/outputs \
    --use_lora true \
    # --use_deepspeed true \
    # --bf16 true \
    # --fp16 false \
    # 
    # --num_train_epochs 1 \
    # --per_device_train_batch_size 2 \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 16 \
    # --evaluation_strategy "no" \
    # --save_strategy "epoch" \
    # --save_total_limit 3 \
    # --learning_rate 4e-4 \
    # --logging_steps 10 \
    # --tf32 False \
    # --model_max_length 512
