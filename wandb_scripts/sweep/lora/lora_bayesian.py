from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftType
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import csv
import wandb
import os

sweep_config = {
    'name': 'lora_sweep_bayesians',
    'method': 'bayes'
    }

metric = {
    'name': 'eval_epoch_loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'r': {
        'values': [4, 8, 64]
    },
    'lora_dropout': {
        'values': [0.001, 0.01, 0.1]
    },
    'lora_alpha': {
        'values': [16, 32, 64]
    },
    'batch_size': {
        'values': [4, 6, 8]
    },
    'lr': {
        'values': [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    }
}

parameters_dict.update(
    {
        # comment epochs to start a real sweep
        'epochs': {'value': 30},
        'max_length': {'value': 128},
        'model_path': {'value': os.environ.get('MODEL_PATH')},
        'save_path': {'value': os.environ.get('SAVE_PATH')},
        'dataset_path': {'value': os.environ.get('DS_PATH')},
        'target_modules': {'value': ["q_proj", "v_proj"]},
        'bias': {'value': "none"}
    }
)
device = "cuda:0"

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="llama-2-7b-peft")


def train(config=None):
    with wandb.init(config=config):
        config=wandb.config
        peft_model_id = f"lora_b{config.batch_size}_e{config.epochs}_lr{str(config.lr)}_maxl{config.max_length}_dp{config.lora_dropout}_al{config.lora_alpha}_r{config.r}"
        print(peft_model_id)
        model_name_or_path = config.model_path

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=config.r, 
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias=config.bias
        )
        text_column = "question"
        label_column = "expected_fields"

        max_length = config.max_length
        lr = config.lr
        num_epochs = config.epochs
        batch_size = config.batch_size

        if config.get("use_hf_data"):
            dataset = load_dataset(config.dataset_path)

        else:
            # dataset from from local path
            def get_successful(all_data):
                successful=[]
                for obj in all_data:
                    if type(obj[label_column]) == list:
                        obj[label_column] = str(obj[label_column])

                    # handling for old dataset with quality column
                    if "quality" in obj:
                        if obj["quality"] == "1" or obj["quality"] == "2":
                            successful.append(obj)

                    # new dataset without quality column
                    else:
                        successful.append(obj)
                return successful

            if ".csv" in config.dataset_path:
                with open(config.dataset_path, "r") as f:
                    csv_reader = csv.DictReader(f)
                    successful = get_successful(csv_reader)

            elif ".json" in config.dataset_path:
                with open(config.dataset_path, "r") as f:
                    qa_pairs = json.load(f)
                    successful = get_successful(qa_pairs)
            else:
                raise ValueError("unsupported dataset file format.")

            dataset = Dataset.from_list(successful)

        # TODO: change to a separate test data set
        dataset = dataset.train_test_split(test_size=0.1)

        print(dataset)

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
            wandb.log({
                "epoch": epoch,
                "train_ppl": train_ppl,
                "train_epoch_loss": train_epoch_loss,
                "eval_ppl": eval_ppl,
                "eval_epoch_loss": eval_epoch_loss
            })
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")



        model.eval()
        i = 8
        inputs = tokenizer(f'{text_column} : {dataset["test"][i][text_column]} Label : ', return_tensors="pt")
        print(dataset["test"][i][text_column])
        print(inputs)
        
        
        save_path = f"{config.save_path}/{peft_model_id}"
        print(f"saving model to {save_path}")
        model.save_pretrained(save_path)
        print("saved")
        
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=config['max_length'], eos_token_id=2
            )
            print(outputs)
            print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

        print("saved")

if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=10)