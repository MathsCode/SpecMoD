import pickle
file_path = '/home/xujiaming/xujiaming/Paper/NIPS_SpecMoD/result/cal_sim/sim_data.pkl'
with open(file_path, 'rb') as file:
    loaded_list = pickle.load(file)

import torch
all_data = torch.tensor(loaded_list)

for i in range(all_data.shape[1]):
    data_token = all_data[:, i, 0]
    