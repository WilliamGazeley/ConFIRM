from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import get_peft_model, TaskType, AdaptionPromptConfig
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Tuple
import ast
import os
import json
import csv
import wandb

testing_targets = {
    "adapt" : {
        "99" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_adapt_b4_e50_lr0.01_maxl128_alen10_alay8_99n_ep6",
        "198" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_adapt_b4_e50_lr0.01_maxl128_alen10_alay8_198n_ep14",
        "495" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_adapt_b4_e50_lr0.01_maxl128_alen10_alay8_495n_ep10",
        "1485" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_adapt_b4_e50_lr0.01_maxl128_alen10_alay8_1485n_ep21",
        "2970" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_adapt_b4_e50_lr0.01_maxl128_alen10_alay8_2970n_ep38",
    },
    "lora" : {
        "99" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_lora_b8_e50_lr0.0003_maxl128_dp0.001_al64_r4_99n_ep4",
        "198" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_lora_b8_e50_lr0.0003_maxl128_dp0.001_al64_r4_198n_ep4",
        "495" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_lora_b8_e50_lr0.0003_maxl128_dp0.001_al64_r4_495n_ep6",
        "1485" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_lora_b8_e50_lr0.0003_maxl128_dp0.001_al64_r4_1485n_ep12",
        "2970" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_lora_b8_e50_lr0.0003_maxl128_dp0.001_al64_r4_2970n_ep24",
    },
    "prefix" : {
        "99" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_prefix_b4_e50_lr0.001_maxl128_nvt30_99n_ep12",
        "198" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_prefix_b4_e50_lr0.001_maxl128_nvt30_198n_ep15",
        "495" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_prefix_b4_e50_lr0.001_maxl128_nvt30_495n_ep9",
        "1485" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_prefix_b4_e50_lr0.001_maxl128_nvt30_1485n_ep16",
        "2970" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_prefix_b4_e50_lr0.001_maxl128_nvt30_2970n_ep19",
    },
    "ptune" : {
        "99" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_ptune_b3_e50_lr0.003_maxl128_nvt15_ehs64_99n_ep10",
        "198" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_ptune_b3_e50_lr0.003_maxl128_nvt15_ehs64_198n_ep15",
        "495" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_ptune_b3_e50_lr0.003_maxl128_nvt15_ehs64_495n_ep10",
        "1485" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_ptune_b3_e50_lr0.003_maxl128_nvt15_ehs64_1485n_ep21",
        "2970" : "asklora_ml_models/peft/llama_2_7b/tuned/ocean_masked_ptune_b3_e50_lr0.003_maxl128_nvt15_ehs64_2970n_ep34",
    }
}

sweep_config = {
    'name': 'model_testing_masked',
    'method': 'grid'
}

metric = {
    'name': 'accuracy',
    'goal': 'minimize'   
}

sweep_config['metric'] = metric
parameters_dict = {
    "method" : {"values": ["adapt", "lora", "prefix", "ptune"]},
    "training_size": {"values": ["99", "198", "495", "1485", "2970"]},
    'peft_model_path_prefix': {'value': os.environ.get('SAVE_PATH')},
    'llama_model_path': {'value': os.environ.get('MODEL_PATH')},
    "testset_path": {"value": "datasets/ocean/descriptive/ocean_rephrased_validated_masked_1000n_test.csv"}
}

device = "cuda"
sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="llama-2-7b-peft")

def success_function_single(
        retrieved_fields: str, 
        expected_output: str
    ) -> Tuple[bool, float]:
    """
    Adaptor for evaluation function
    
    Args:
        retrieved_fields: List of retrieved fields
        expected_output: List of expected fields in format table.field
​
    Returns:
        success: boolean indicating whether retrieved fields match expected output
        jaccard_distance: float indicating the Jaccard distance between retrieved and expected fields
    """  
    if isinstance(retrieved_fields, str):
        try:
            retrieved_fields = ast.literal_eval(retrieved_fields)
        except Exception as e:
            retrieved_fields = []
    if isinstance(expected_output, str):
        try:
            expected_output = ast.literal_eval(expected_output)
        except Exception as e:
            expected_output = []

    expected_output_str = ' '.join(expected_output)
    
    # If the external_data categorisation is wrong, fail immediately
    if ("external_data" in expected_output_str and "external_data" not in retrieved_fields) or \
        ("external_data" not in expected_output_str and "external_data" in retrieved_fields):
        return False, 0

    # Calculate intersection and union
    intersection = len([f for f in retrieved_fields if f in expected_output])
    union = len(set(retrieved_fields)) + len(set(expected_output)) - intersection

    # Calculate Jaccard distance
    jaccard_distance = 1 - intersection / union if union != 0 else 0

    if all([f in retrieved_fields for f in expected_output]):
        return True, jaccard_distance
    else:
        return False, jaccard_distance

def test(config=None):
    with wandb.init(config=config):
        config=wandb.config
        model_id = testing_targets[config.method][config.training_size]
        model_name = model_id.split("tuned")[1]
        model_path = config.peft_model_path_prefix + model_name
        
        text_column = "question"
        label_column = "expected_fields"
        
        successful=[]
        def get_successful(all_data):
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

        if ".csv" in config.testset_path:
            with open(config.testset_path, "r") as f:
                csv_reader = csv.DictReader(f)
                successful = get_successful(csv_reader)

        elif ".json" in config.testset_path:
            with open(config.testset_path, "r") as f:
                qa_pairs = json.load(f)
                successful = get_successful(qa_pairs)

        else:
            raise ValueError("unsupported dataset file format.")

        testset = Dataset.from_list(successful)
        
        tokenizer = LlamaTokenizer.from_pretrained(config.llama_model_path)
        
        model = LlamaForCausalLM.from_pretrained(config.llama_model_path)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.to(device)
        model.eval()

        test_results = []

        with torch.no_grad():
            for test_obj in testset:
            
                input_text = f"{test_obj[text_column]} Label : "
                inputs = tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,
                    eos_token_id=2
                )
                output_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

                test_result = success_function_single(output_text.replace(input_text, '').strip(), test_obj[label_column])

                print(f"output: {output_text}")

                print(f"expected: {test_obj[label_column]}")
                print(test_result)

                test_results.append(test_result)
        
        t = 0
        f = 0
        total_score = 0

        for success, score in test_results:
            if success: t += 1
            else: f += 1
            total_score += score

        print(f"accuracy: {100 * t / (t + f)}")
        print(f"avg jscore: {total_score / len(test_results)}")
        wandb.log({
            "method": config.method,
            "train_size": config.training_size,
            "t": t,
            "f": f,
            "accuracy": t / (t + f),
            "avg_jscore": total_score / len(test_results)
        })
        # clean up
        del tokenizer
        del model
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    wandb.agent(sweep_id, test)