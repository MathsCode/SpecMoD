from transformers import AutoTokenizer, AutoModelForCausalLM
from model.EAGLE_model import Model as SpecModel
import torch
from typing import Optional, Union


import torch
def main(args):
    model_name = args.model
    dataset = args.dataset
    begin = 'None'
    end = 'None'

    import json
    with open(f"./train_data/{dataset}_{model_name}_data_{begin}_{end}.json", "r") as f:
        data = json.load(f)

    Ori_model_path = f"/inspire/hdd/global_public/public_models/Qwen/{model_name}/"
    if model_name == 'Qwen3-8B':
        Spec_model_path = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/qwen3_8b_eagle3"
    elif model_name == 'Qwen3-14B':
        Spec_model_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/Qwen3-14B_eagle3'
    else:
        assert "None Support"
    tokenizer = AutoTokenizer.from_pretrained(Ori_model_path)
    ori_model = AutoModelForCausalLM.from_pretrained(Ori_model_path, device_map='auto', torch_dtype=torch.float16)

    N_LAYERS = ori_model.config.num_hidden_layers  
    HIDDEN_DIM = ori_model.config.hidden_size


    spec_model = SpecModel.from_pretrained(Spec_model_path=Spec_model_path, Ori_model_path=Ori_model_path, dtype=torch.float16).to(ori_model.device)

    cur_hidden_states = torch.load(f'./train_data/{dataset}_{model_name}_cur_hidden_states_{begin}_{end}.pt').to(ori_model.device)
    fake_last_hidden_states = torch.load(f'./train_data/{dataset}_{model_name}_fake_last_hidden_states_{begin}_{end}.pt').to(ori_model.device)
    true_last_hidden_states = torch.load(f'./train_data/{dataset}_{model_name}_true_last_hidden_states_{begin}_{end}.pt').to(ori_model.device)



    train_data_last_hidden_states = []
    train_data_cur_hidden_states = []
    train_data_spec_hidden_states = []
    train_label = []
    start_idx = 0
    for key, value in data.items():
        prompt = value['Prompt']
        input_ids = torch.tensor([prompt]).to(ori_model.device)
        outputs = ori_model.model(
            input_ids=input_ids,
            use_cache=True,
        )
        
        last_hidden_state = outputs.last_hidden_state
        orig = ori_model.lm_head(last_hidden_state[:, -1])
        token = torch.argmax(orig)
        record_input_id = value['Token'][0]['input_id']
        token = token[None, None]
        assert token.item() == record_input_id, "Something wrong in Prefill"
        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        spec_model.reset_kv()
        spec_hidden_states = spec_model.topK_genrate(
            hidden_states=last_hidden_state,
            input_ids=input_ids)
        train_data_spec_hidden_states.append(spec_hidden_states)
        train_label.append(value['Token'][0]['layer_index'])
        for idx in range(1,len(value['Token'])):
            token_json = value['Token'][idx]
            last_hidden_state = true_last_hidden_states[start_idx+idx:start_idx+idx+1]
            input_id = torch.tensor(token_json['input_id']).to(ori_model.device)
            input_id = input_id[None, None]
            input_ids = torch.cat((input_ids, input_id), dim=1)
            # print(input_ids.shape)
            # print(last_hidden_state.shape)
            spec_hidden_states = spec_model.topK_genrate(
                hidden_states=last_hidden_state,
                input_ids=input_ids
            )
            train_data_spec_hidden_states.append(spec_hidden_states)
            train_label.append(token_json['layer_index'])
        start_idx += len(value['Token'])+1
    assert start_idx == true_last_hidden_states.shape[0], "Something wrong in start_idx"
    label_data =  torch.zeros((len(train_label), N_LAYERS))
    for i, layer_idx in enumerate(train_label):
        label_data[i, layer_idx] = 1.0

    train_data_spec_hidden_states = torch.cat(train_data_spec_hidden_states, dim=0)

    # print(spec_hidden_states.shape)
    # print(cur_hidden_states.shape)
    # print(true_last_hidden_states.shape)

    torch.save(train_data_spec_hidden_states,f'./train_data/{dataset}_{model_name}_spec_hidden_states.pt')
    torch.save(label_data, f'./train_data/{dataset}_{model_name}_label_data.pt')

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    args = parser.parse_args()
    main(args)

    
