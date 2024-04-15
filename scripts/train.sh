experiment_name="dia-llama"

CUDA_VISIBLE_DEVICES=1,5 torchrun --nproc_per_node=2 /home/chenzhixuan/Workspace/LLM4CTRG/src/train.py \
    --bf16 True \
    --lang_encoder_path "/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf" \
    --tokenizer_path "/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf" \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --dataloader_num_workers 8 \
    --run_name $experiment_name \
    --output_dir "/data/chenzhixuan/checkpoints/LLM4CTRG/outputs/$experiment_name" \
    --deepspeed "/home/chenzhixuan/Workspace/LLM4CTRG/ds_configs/stage2.json" \
    --logging_steps 1