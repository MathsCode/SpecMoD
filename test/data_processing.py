import argparse
import pickle
import os

def process(args, data, save_path):
    if args.skip_distribution:
        # data preprocess
        data_dist = [0] * args.total_layers
        tot_skip_layer = 0
        tot_tokens = len(data)
        for data_item in data:
            tot_skip_layer += data_item[0]
            for skip_item in data_item[1]:
                data_dist[skip_item] += 1
        # calcualte average skip layers
        print("average skip layer = {}".format(tot_skip_layer / tot_tokens))
        # draw line chart
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(0, args.total_layers)
        y = np.array(data_dist)
        plt.plot(x, y)
        plt.savefig(save_path)
        plt.clf()
        
def main(args):
    if args.file_dir is not None:
        for root, dirs, files in os.walk(args.file_dir):
            for file in files:
                file_path = os.path.join(root,file)
                print(file_path)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                save_path = file_path.replace('pkl','png')
                process(args,data,save_path)
    elif args.file is not None:
        with open(args.file, 'rb') as file:
            data = pickle.load(file)
            save_path = os.path.join(os.path.dirname(args.file), args.save_name)
            process(args,data,save_path)
    else:
        print('Invalid File!')
    
        
   
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, default=None)
    parser.add_argument("--file_dir", "-fd", type=str, default=None)
    parser.add_argument("--save_name", "-n", type=str)
    parser.add_argument("--total_layers", "-l", type=int, default=64)
    parser.add_argument("--skip_distribution", "-skip", action='store_true', default=False)
    parser.add_argument("--sim_distribution", "-sim", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
     