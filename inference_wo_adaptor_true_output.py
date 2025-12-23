'''
This file is mainly designed for collecting training data for the adaptor in each layer.

The details are in Feishu "Adaptor 训练数据"

'''


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
    from model.qwen3_model_train import  Spec_Qwen3ForCausalLM
    
    if args.device == "infini":
        model_path = f"/share/others/public_models/{args.model}/"
        dataset_path = '/home/xujiaming/xujiaming/Paper/dataset/'+args.dataset+'/question.jsonl'
    elif args.device == "qz":
        model_path = f"/inspire/hdd/global_public/public_models/Qwen/{args.model}/"
        dataset_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/benchmark/'+args.dataset+'/question.jsonl'
    else:
        assert False, "device error"
        
    

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Spec_Qwen3ForCausalLM.from_pretrained(model_path).half().to('cuda')
    

    LAYERS = model.config.num_hidden_layers
    train_X = [[] for i in range(LAYERS)]
    train_Y = [[] for i in range(LAYERS)]
    storage.reset()
    save_json = {}
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
        outputs = model.generate(**inputs, max_new_tokens=args.max_gen, do_sample = False, temperature=None, top_p=None, top_k = None)
        json_data, cur_hidden_states, fake_last_hidden_states, true_last_hidden_states, total_length, total_tokens = storage.get_data()
        save_json_item['Token'] = json_data
        save_json_item['avg_len'] = total_length/total_tokens if total_tokens > 0 else 0
        save_json[question["question_id"]] = save_json_item
        train_x, train_y = storage.get_train_data()
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
            torch.save(train_X[i], f'./train_data/adaptor/0.95/{args.dataset}_{args.model}_X_idx{i}_{args.begin}_{args.end}.pt')
            torch.save(train_Y[i], f'./train_data/adaptor/0.95/{args.dataset}_{args.model}_Y_idx{i}_{args.begin}_{args.end}.pt')
    
    save_path = f'./train_data/adaptor/0.95/{args.dataset}_{args.model}_data_{args.begin}_{args.end}.json'
    with open(save_path, "w") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="infini")
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--setting", type=str, default='dev')
    parser.add_argument("--max_gen", type=int, default=100)
    args = parser.parse_args()
    save_path = f'./output/baseline_{args.dataset}_{args.model}_output_{args.begin}_{args.end}_0.95.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        main(args)