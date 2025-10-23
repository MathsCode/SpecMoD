from transformers import AutoTokenizer
from model.qwen3_model import  Spec_Qwen3ForCausalLM
# /share/public/public_models/Qwen3-14B/

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from typing import Optional
import json, tqdm

from model.utils import storage
def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions



def main(args):

    save_json = {}
    storage.reset()

    tokenizer = AutoTokenizer.from_pretrained(f"/share/others/public_models/{args.model}/")
    model = Spec_Qwen3ForCausalLM.from_pretrained(f"/share/others/public_models/{args.model}/")

    model.to("cuda")


    dataset_path = '/home/xujiaming/xujiaming/Paper/dataset/'+args.dataset+'/question.jsonl'
    questions = load_questions(dataset_path,args.begin,args.end)
    messages = [
        {"role": "system",
            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
    ]
    for question in tqdm.tqdm(questions):
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
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.01)
        json_data, total_length, total_tokens = storage.get_data()
        save_json_item['Token'] = json_data
        save_json_item['avg_len'] = total_length/total_tokens
        save_json[question["question_id"]] = save_json_item
        # print()
        # print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
        # print(outputs[0][inputs["input_ids"].shape[-1]:])
    print(save_json)
    save_path = f'./data/{args.dataset}_{args.model}_data_{args.begin}_{args.end}.json'
    with open(save_path, "w") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-14B")
    parser.add_argument("--begin", "-b", type=int,default=0)
    parser.add_argument("--end", "-e", type=int,default=1)
    parser.add_argument("--thinking", "-t", action="store_true", default=False)
    args = parser.parse_args()
    main(args)