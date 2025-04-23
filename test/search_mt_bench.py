from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
from load_question import load_questions,temperature_config
import torch
from tqdm import tqdm
import os
import json

from customized_model.SpecMoD_qwen2 import Qwen2ForCausalLM

from transformers.generation.utils import GenerationMixin

def main(args):
    if args.model == 'QwQ':
        model_name = "/home/xujiaming/xujiaming/models/QwQ-32B"
        model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model_name = args.model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    # elif args == 'Llama3-8B':
    #     model_name = "/share/public/public_models/Llama3-8B"
    
    
    
    dataset_path = '/home/xujiaming/xujiaming/Paper/NIPS_SpecMoD/dataset/'+args.dataset+'/question.jsonl'
    if args.search:
        save_file = "/home/xujiaming/xujiaming/Paper/NIPS_SpecMoD/result/search/{}_{}_{}.jsonl".format(args.model,args.dataset,args.skip_layers)
    else:
        save_file = "/home/xujiaming/xujiaming/Paper/NIPS_SpecMoD/result/answer/{}_{}.jsonl".format(args.model,args.dataset)
    questions = load_questions(dataset_path,args.begin,args.end)
    

    # prompt = "How many r's are in the word \"strawberry\""
    
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        
        # cancel the dynamic output
        if args.search:
            temperature = 0
            
        prompt = question["turns"][0]
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            # temperature=temperature,
            Spec_search = args.search,
            do_sample = False,
        )
        # max_new_tokens = 32768
        # for i in range(max_new_tokens):
            

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, "a") as f_save:
            ans_json ={
                "question_id": question["question_id"],
                "response": response
            }
            f_save.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--begin", "-b", type=int,default=0)
    parser.add_argument("--end", "-e", type=int,default=1)
    parser.add_argument("--search", "-s", action='store_true', default=False)
    parser.add_argument("--skip_layers","-SL", type=int, default=0)
    parser.add_argument("--max_new_tokens", "-n", type=int, default=256)
    args = parser.parse_args()
    main(args)