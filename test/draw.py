import pickle
file_path = '/home/xujiaming/xujiaming/Paper/NIPS_SpecMoD/result/cal_sim/sim_data.pkl'
with open(file_path, 'rb') as file:
    loaded_list = pickle.load(file)
    
import torch
sim_data = torch.tensor(loaded_list)
import numpy as np
import matplotlib.pyplot as plt
for i in range(sim_data.shape[0]):
    sim_layer = sim_data[i,:,0].numpy()
    plt.boxplot(sim_layer)
    plt.title(f"Layer {i+1}")
    plt.savefig(f"layer_{i+1}.png")
    plt.clf()
    