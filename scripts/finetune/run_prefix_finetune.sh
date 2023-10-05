python scripts/finetune/llama_2_prefix_tune.py \
 --model_path $MODEL_PATH \
 --save_path $SAVE_PATH \
 --num_virtual_tokens 30 \
 --batch_size 8 \
 --epochs 50 \
 --lr 0.01 \
 --max_length 128 \
 --dataset_path $DS_PATH