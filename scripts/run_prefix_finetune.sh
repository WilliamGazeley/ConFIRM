python llama_2_prefix_tune.py \
 --model_path /home/adrianw/everything/nlp/model-serving-sandbox/llama/llama-2-7b/hf \
 --save_path /home/adrianw/everything/nlp/model-serving-sandbox/llama/llama-2-7b/tuned \
 --num_virtual_tokens 30 \
 --batch_size 4 \
 --epochs 20 \
 --lr 0.0003 \
 --max_length 128 \
 --dataset_path /home/adrianw/everything/final/ConFIRM/datasets/chagpt/ConFIRM_QAset_gpt_filtered.csv