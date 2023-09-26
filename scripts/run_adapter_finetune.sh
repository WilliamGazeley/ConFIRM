python llama_2_lora_tune.py \
 --model_path /home/adrianw/hf \
 --save_path /home/adrianw/tuned \
 --adapter_len 10 \
 --adapter_layers 30 \
 --batch_size 8 \
 --epochs 50 \
 --lr 0.0003 \
 --max_length 64 \
 --dataset_path /home/adrianw/ConFIRM/datasets/mixed/ConFIRM_QAset_559n_mixed_train.csv
