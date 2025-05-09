import os
import pickle
import argparse


def main(args):
    # root_path = "/home/xujiaming/xujiaming/Paper/NIPS_SpecMoD/result/layer_data/"
    # dataset = "alpaca"
    
    all_data = []
    for i in range(args.total_number):
        file_name = "{}_question_{}.pkl".format(args.dataset, i)
        file_path = os.path.join(args.file_dir, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_data += data
    result_dict = {}
    union_result_dict = {}
    inter_result_dict = {}
    for data_item in all_data:
        input_id, output_id, length, skip_layer = data_item
        key = (input_id, output_id)
        if key in result_dict:
            result_dict[key].append(skip_layer)
        else:
            result_dict[key] = [skip_layer]
    
    for key, values in result_dict.items():
        union_result_dict[key] = set()
        inter_result_dict[key] = {i for i in range(64)}
        for value in values:
            union_result_dict[key] = set(value) | union_result_dict[key]
            inter_result_dict[key] = inter_result_dict[key] & set(value)
        print(key, union_result_dict[key],len(union_result_dict[key]))
        print(key, inter_result_dict[key],len(inter_result_dict[key]))
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", "-fd", type=str, default=None)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--total_number", "-t", type=int, default=6)
    args = parser.parse_args()
    main(args)