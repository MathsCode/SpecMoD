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

from model.utils import storage, ShadowAdapter2,PathPredictorMLP
from model.EAGLE_model import Model as SpecModel

from model.utils import Spec_update_model_kwargs_for_generation

from transformers.generation.utils import GenerationMixin
GenerationMixin._update_model_kwargs_for_generation = Spec_update_model_kwargs_for_generation

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
    from model.qwen3_model_adaptor_router_pipeline import  Spec_Qwen3ForCausalLM
    
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
    HIDDEN_DIM = model.config.hidden_size
    MLP_INTERNAL_DIM = 2048
    adaptor = []
    for i in range(LAYERS):
        layer_adaptor = ShadowAdapter2(model.config.hidden_size, 2048)
        layer_adaptor_weight = torch.load(f"./checkpoint/adaptor/2048/adapter_layer_{i}_2048_Model2.pt")
        layer_adaptor.load_state_dict(layer_adaptor_weight)
        layer_adaptor = layer_adaptor.half().to(model.device)
        adaptor.append(layer_adaptor)
    if args.model == 'Qwen3-8B':
        Spec_model_path = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/qwen3_8b_eagle3"
    elif args.model == 'Qwen3-14B':
        Spec_model_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/Qwen3-14B_eagle3'

    spec_model = SpecModel.from_pretrained(Spec_model_path=Spec_model_path, Ori_model_path=model_path, dtype=torch.float16).to(model.device)
    router = PathPredictorMLP(LAYERS, HIDDEN_DIM, MLP_INTERNAL_DIM)
    
    router.load_state_dict(torch.load(f"./checkpoint/router/{args.model}_layer_router_{MLP_INTERNAL_DIM}.pth"))
    router = router.half().to(model.device)
    
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
        spec_model.reset_kv()
        outputs = model.generate(**inputs, max_new_tokens=args.max_gen, temperature=1, adaptor=adaptor, spec_model=spec_model, router=router, last_hidden_state = None )
        
        print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    import argparse 
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
    main(args)