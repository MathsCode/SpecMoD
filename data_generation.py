from transformers import AutoTokenizer

# /share/public/public_models/Qwen3-14B/


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
    
    if args.dev:
        from model.qwen3_model_dev import  Spec_Qwen3ForCausalLM
        
        from transformers.generation.utils import ALL_CACHE_NAMES
        ALL_CACHE_NAMES.extend(['past_hidden_states'])

        from model.utils import Spec_update_model_kwargs_for_generation

        from transformers.generation.utils import GenerationMixin
        GenerationMixin._update_model_kwargs_for_generation = Spec_update_model_kwargs_for_generation
    elif args.profile:
        from model.qwen3_model_profile import  Spec_Qwen3ForCausalLM
    
    if args.device == "infini":
        model_path = f"/share/others/public_models/{args.model}/"
        dataset_path = '/home/xujiaming/xujiaming/Paper/dataset/'+args.dataset+'/question.jsonl'
    elif args.device == "qz":
        model_path = f"/inspire/hdd/global_public/public_models/Qwen/{args.model}/"
        dataset_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/benchmark/'+args.dataset+'/question.jsonl'
    else:
        assert False, "device error"
        
    save_json = {}
    save_cur_hidden_states = []
    save_true_last_hidden_states = []
    save_fake_last_hidden_states = []
    storage.reset()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Spec_Qwen3ForCausalLM.from_pretrained(model_path).half()

    model.to("cuda")

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
        if args.dev:
            outputs = model.generate(**inputs, max_new_tokens=args.max_gen, temperature=0.01, use_buffer=True, past_hidden_states=None)
        if args.profile:
            outputs = model.generate(**inputs, max_new_tokens=args.max_gen, temperature=0.01)
            json_data, cur_hidden_states, fake_last_hidden_states, true_last_hidden_states, total_length, total_tokens = storage.get_data()
            save_json_item['Token'] = json_data
            save_json_item['avg_len'] = total_length/total_tokens if total_tokens > 0 else 0
            save_json[question["question_id"]] = save_json_item
            cur_hidden_states = torch.cat(cur_hidden_states, dim=0).cpu()
            fake_last_hidden_states = torch.cat(fake_last_hidden_states, dim=0).cpu()
            true_last_hidden_states = torch.cat(true_last_hidden_states, dim=0).cpu()
            print(cur_hidden_states.shape)
            print(fake_last_hidden_states.shape)
            print(true_last_hidden_states.shape)
            save_cur_hidden_states.append(cur_hidden_states)
            save_fake_last_hidden_states.append(fake_last_hidden_states)
            save_true_last_hidden_states.append(true_last_hidden_states)
            storage.reset()
        
        # print()
        # print(tokenizer.decode(outputs[0]))
        # print(outputs[0][inputs["input_ids"].shape[-1]:])
    if args.profile:
        save_cur_hidden_states = torch.cat(save_cur_hidden_states, dim=0)
        save_fake_last_hidden_states = torch.cat(save_fake_last_hidden_states, dim=0)
        save_true_last_hidden_states = torch.cat(save_true_last_hidden_states, dim=0)
        torch.save(save_cur_hidden_states, f'./train_data/{args.dataset}_{args.model}_cur_hidden_states_{args.begin}_{args.end}.pt')
        torch.save(save_fake_last_hidden_states, f'./train_data/{args.dataset}_{args.model}_fake_last_hidden_states_{args.begin}_{args.end}.pt')
        torch.save(save_true_last_hidden_states, f'./train_data/{args.dataset}_{args.model}_true_last_hidden_states_{args.begin}_{args.end}.pt')
        
        print(save_json)
        save_path = f'./train_data/{args.dataset}_{args.model}_data_{args.begin}_{args.end}.json'
        with open(save_path, "w") as f:
            json.dump(save_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="infini")
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--max_gen", type=int, default=100)
    args = parser.parse_args()
    main(args)