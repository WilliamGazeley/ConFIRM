from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType, LoraConfig, AdaptionPromptConfig
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import csv
import os
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description=\
    "llama 2 adapter tuning script for ConFIRM using peft" \
    )
# Required arguments
parser.add_argument('--model_path', type=str, required=True,
                    help='path to the model and tokenizer (huggingface format)')

parser.add_argument('--save_path', type=str, required=True,
                    help='save path of the tuned model (peft)')

parser.add_argument('--adapter_len', type=int, required=True,
                    help='Number of adapter tokens to insert')
parser.add_argument('--adapter_layers', type=float, required=True,
                    help='Number of adapter layers (from the top)')
parser.add_argument('--batch_size', type=int, required=True,
                    help='batch size of the training')
parser.add_argument('--max_length', type=int, required=True,
                    help='Max number of tokens generated, range in [0, 2048], higher max length will result in higher memory requirement')
parser.add_argument('--epochs', type=int, required=True,
                    help='Number of epochs')
parser.add_argument('--lr', type=float, required=True,
                    help='Learning rate')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='dataset path, should point to a file with json format containing a list of object data.')
parser.add_argument('--use_hf_data', 
                    help='whether to use custom dataset. If set, dataset_path will automatically point to a hf dataset repository')



def main(**kwargs):
    
    device = "cuda"
    model_name_or_path = kwargs["model_path"]
    peft_config = AdaptionPromptConfig(
        task_type=TaskType.CAUSAL_LM,
        adapter_layers=kwargs["adapter_layers"],
        adapter_len=kwargs["adapter_len"],
    )
    
    text_column = "question"
    label_column = "expected_fields"

    max_length = kwargs['max_length']
    lr = kwargs['lr']
    num_epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    
    if kwargs.get("use_hf_data"):
        dataset = load_dataset(kwargs['dataset_path'])
        
    else:
        # dataset from from local path
        def get_successful(all_data):
            successful=[]
            for obj in all_data:
                if type(obj[label_column]) == list:
                    obj[label_column] = str(obj[label_column])
                else:
                    successful.append(obj)
            return successful
                    
        if ".csv" in kwargs['dataset_path']:
            with open(kwargs['dataset_path'], "r") as f:
                csv_reader = csv.DictReader(f)
                successful = get_successful(csv_reader)

        elif ".json" in kwargs['dataset_path']:
            with open(kwargs['dataset_path'], "r") as f:
                qa_pairs = json.load(f)
                successful = get_successful(qa_pairs)
        else:
            raise ValueError("unsupported dataset file format.")

        dataset = Dataset.from_list(successful)
    
    # Makes for prettier splits
    dataset = dataset.train_test_split(train_size=round(len(dataset) / 1.1))
    
    print(dataset)

    peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
    print(peft_model_id)

    # data preprocessing
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    
    # llama does not have pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]


    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


    # creating model
    model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(model.peft_config)


    # model
    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )


    # training and evaluation
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            #         print(batch)
            #         print(batch["input_ids"].shape)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")



    model.eval()
    i = 8
    inputs = tokenizer(f'{text_column} : {dataset["test"][i][text_column]} Label : ', return_tensors="pt")
    print(dataset["test"][i][text_column])
    print(inputs)

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=kwargs['max_length'], eos_token_id=2
        )
        print(outputs)
        print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


    print(f"saving model to {kwargs['save_path']}")
    model.save_pretrained(kwargs['save_path'])
    print("saved")

    ckpt = f"{kwargs['save_path']}/adapter_model.bin"

if __name__=="__main__":
    
    args = parser.parse_args()
    main(**vars(args))