
CUDA_VISIBLE_DEVICES=0 python data_generation.py -d mt-bench &
CUDA_VISIBLE_DEVICES=1 python data_generation.py -d gsm8k &
CUDA_VISIBLE_DEVICES=2 python data_generation.py -d alpaca &
CUDA_VISIBLE_DEVICES=3 python data_generation.py -d sum &
CUDA_VISIBLE_DEVICES=4 python data_generation.py -d vicuna-bench &
CUDA_VISIBLE_DEVICES=5 python data_generation.py -d math_infini