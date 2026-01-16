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
import torch.nn as nn
from model.utils import storage, ShadowAdapter2, ShadowAdapter3, PathPredictorMLP, record, Global_router
import time
from model.EAGLE_model import Model as SpecModel
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
    from model.qwen3_model_adaptor_global_router_pipeline import  Spec_Qwen3ForCausalLM
    from model.utils import Spec_update_model_kwargs_for_generation
    from transformers.generation.utils import GenerationMixin
    GenerationMixin._update_model_kwargs_for_generation = Spec_update_model_kwargs_for_generation
    model_path = f"/inspire/hdd/global_public/public_models/Qwen/{args.model}/"
    dataset_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/benchmark/'+args.dataset+'/question.jsonl'

    

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ori_model = Spec_Qwen3ForCausalLM.from_pretrained(model_path).half().to('cuda')
    Spec_model_path = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/qwen3_8b_eagle3"
    spec_model = SpecModel.from_pretrained(Spec_model_path=Spec_model_path, Ori_model_path=model_path, dtype=torch.float16).to(ori_model.device)
    LAYERS = ori_model.config.num_hidden_layers
    adaptor = [None, ]
    router = Global_router(input_dim=ori_model.config.hidden_size*2, hidden_dim=1024, output_dim=LAYERS).to(ori_model.device)
    router_weight = torch.load("/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/global_router/global_router_1024_Model1_non_thinking_first.pt")
    # router = PathPredictorMLP(n_layers=LAYERS, mlp_internal_dim=2048, llm_hidden_dim=ori_model.config.hidden_size*2).to(ori_model.device)
    # router_weight = torch.load("/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/global_router/global_router_2048_Model2.pt")
    router.load_state_dict(router_weight)
    router = router.half()
    
    
    
    for i in range(1, LAYERS):
        if i == 34:
            adaptor.append(None)
        else:
            layer_adaptor = ShadowAdapter3(ori_model.config.hidden_size, 1024)
            layer_adaptor_weight = torch.load(f"./checkpoint/adaptor/1024/adapter_layer_{i}_1024_Model3_0.95.pt")
            layer_adaptor.load_state_dict(layer_adaptor_weight)
            layer_adaptor = layer_adaptor.half().to(ori_model.device)
            adaptor.append(layer_adaptor)
    

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
            enable_thinking = False, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(ori_model.device)

        outputs = ori_model.generate(**inputs, max_new_tokens=args.max_gen, temperature=0.000001, adaptor=adaptor, router=router,spec_model=spec_model, last_hidden_state = None)
        print(tokenizer.decode(outputs[0]))
        
        print(record.get_average_len())
    
  

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--max_gen", type=int, default=100)
    args = parser.parse_args()
    main(args)