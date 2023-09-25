python llama_2_lora_tune.py \
 --model_path /home/adrianw/everything/nlp/model-serving-sandbox/llama/llama-2-7b/hf \
 --save_path /home/adrianw/everything/nlp/model-serving-sandbox/llama/llama-2-7b/tuned \
 --r 8 \
 --lora_dropout 0.05 \
 --lora_alpha 32 \
 --batch_size 4 \
 --epochs 10 \
 --lr 0.0001 \
 --max_length 128 \
 --dataset_path /home/adrianw/everything/final/ConFIRM/datasets/chagpt/ConFIRM_QAset_gpt_filtered.csv