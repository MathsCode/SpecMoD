from transformers import AutoTokenizer

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from typing import Optional
import json, tqdm
import torch

from model.utils import storage



def load_questions(question_file: str, begin=None, end=None):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions



def main(args):
    from model.llama_base_model_find_path import  Spec_LlamaForCausalLM
    
    model_path = f"/inspire/hdd/global_public/public_models/meta-llama/Llama-3.1-8B-Instruct/"
    dataset_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/benchmark/'+args.dataset+'/question.jsonl'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Spec_LlamaForCausalLM.from_pretrained(model_path).half().to('cuda')
    LAYERS = model.config.num_hidden_layers
    
    train_X = [[] for i in range(LAYERS)]
    train_Y = [[] for i in range(LAYERS)]
    storage.reset()
    save_json = {}
    save_last_hidden_states = []
    questions = load_questions(dataset_path,args.begin,args.end)
    for question in tqdm.tqdm(questions):
        messages = [
            {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        prompt = question["turns"][0]
        messages.append({
            "role": "user",
            "content": prompt
        })
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        save_json_item = {"Prompt":inputs.input_ids.squeeze(0).tolist()}
        outputs = model.generate(**inputs, max_new_tokens=args.max_gen, temperature=0.0000001)
        json_data, total_length, total_tokens = storage.get_normal_info()
        save_json_item['Token'] = json_data
        save_json_item['avg_len'] = total_length/total_tokens if total_tokens > 0 else 0
        save_json_item['output'] = outputs[0].cpu().tolist()
        save_json[question["question_id"]] = save_json_item
        train_x, train_y = storage.get_layer_hidden_states()
        last_hidden_states = storage.get_last_hidden_states()
        last_hidden_states = torch.cat(last_hidden_states, dim = 1)
        save_last_hidden_states.append(last_hidden_states)
        
        
        for i in range(LAYERS):
            if len(train_x) > i and train_x[i] != []:
                train_x[i] = torch.cat(train_x[i], dim= 0)
                train_y[i] = torch.cat(train_y[i], dim = 0)
                train_X[i].append(train_x[i])
                train_Y[i].append(train_y[i])
        storage.reset()
        print(tokenizer.decode(outputs[0]))
        
    for i in range(LAYERS):
        if train_X[i] != []:
            train_X[i] = torch.cat(train_X[i], dim = 0)
            train_Y[i] = torch.cat(train_Y[i], dim = 0)
            torch.save(train_X[i], f'./train_data/adaptor/0.9/{args.dataset}_Llama3.1-8B-Instruct_X_idx{i}_{args.begin}_{args.end}.pt')
            torch.save(train_Y[i], f'./train_data/adaptor/0.9/{args.dataset}_Llama3.1-8B-Instruct_Y_idx{i}_{args.begin}_{args.end}.pt')
    save_last_hidden_states = torch.cat(save_last_hidden_states, dim = 1).cpu()
    torch.save(save_last_hidden_states, f'./train_data/global_router/llama/{args.dataset}_Llama3.1-8B-Instruct_last_hidden_states_{args.begin}_{args.end}.pt')
    save_path = f'./train_data/adaptor/0.9/{args.dataset}_Llama3.1-8B-Instruct_data_{args.begin}_{args.end}.json'
    with open(save_path, "w") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=4)
            
            
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--max_gen", type=int, default=100)
    args = parser.parse_args()
    save_path = f'./output/baseline_{args.dataset}_Llama3.1-8B-Instruct_output_{args.begin}_{args.end}_0.9.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        main(args)