CUDA_VISIBLE_DEVICES=0 python data_generation.py --device qz  --profile --max_gen 1000 -d mt-bench -m Qwen3-14B
CUDA_VISIBLE_DEVICES=1 python data_generation.py --device qz  --profile --max_gen 1000 -d gsm8k -m Qwen3-14B
CUDA_VISIBLE_DEVICES=2 python data_generation.py --device qz  --profile --max_gen 1000 -d alpaca -m Qwen3-14B
CUDA_VISIBLE_DEVICES=3 python data_generation.py --device qz  --profile --max_gen 1000 -d sum -m Qwen3-14B
CUDA_VISIBLE_DEVICES=4 python data_generation.py --device qz  --profile --max_gen 1000 -d vicuna-bench -m Qwen3-14B
CUDA_VISIBLE_DEVICES=5 python data_generation.py --device qz  --profile --max_gen 1000 -d math_infini -m Qwen3-14B
CUDA_VISIBLE_DEVICES=6 python data_generation.py --device qz  --profile --max_gen 1000 -d mt-bench -m Qwen3-8B
CUDA_VISIBLE_DEVICES=6 python data_generation.py --device qz  --profile --max_gen 1000 -d gsm8k -m Qwen3-8B
CUDA_VISIBLE_DEVICES=7 python data_generation.py --device qz  --profile --max_gen 1000 -d alpaca -m Qwen3-8B
CUDA_VISIBLE_DEVICES=7 python data_generation.py --device qz  --profile --max_gen 1000 -d sum -m Qwen3-8B
CUDA_VISIBLE_DEVICES=1 python data_generation.py --device qz  --profile --max_gen 1000 -d vicuna-bench -m Qwen3-8B
CUDA_VISIBLE_DEVICES=2 python data_generation.py --device qz  --profile --max_gen 1000 -d math_infini -m Qwen3-8B

CUDA_VISIBLE_DEVICES=0 python train_fuse.py -d alpaca  -m Qwen3-8B
CUDA_VISIBLE_DEVICES=0 python train_fuse.py -d alpaca  -m Qwen3-14B
CUDA_VISIBLE_DEVICES=1 python train_fuse.py -d gsm8k  -m Qwen3-8B
CUDA_VISIBLE_DEVICES=2 python train_fuse.py -d gsm8k  -m Qwen3-14B
CUDA_VISIBLE_DEVICES=3 python train_fuse.py -d math_infini  -m Qwen3-8B
CUDA_VISIBLE_DEVICES=4 python train_fuse.py -d math_infini  -m Qwen3-14B
CUDA_VISIBLE_DEVICES=5 python train_fuse.py -d mt-bench  -m Qwen3-8B
CUDA_VISIBLE_DEVICES=6 python train_fuse.py -d mt-bench  -m Qwen3-14B
CUDA_VISIBLE_DEVICES=7 python train_fuse.py -d sum  -m Qwen3-8B
CUDA_VISIBLE_DEVICES=0 python train_fuse.py -d sum  -m Qwen3-14B
CUDA_VISIBLE_DEVICES=1 python train_fuse.py -d vicuna-bench  -m Qwen3-8B
CUDA_VISIBLE_DEVICES=2 python train_fuse.py -d vicuna-bench  -m Qwen3-14B



CUDA_VISIBLE_DEVICES=1 python data_generation.py --device qz  --profile --max_gen 1000 -d mt-bench -m Qwen3-8B
CUDA_VISIBLE_DEVICES=2 python data_generation.py --device qz  --profile --max_gen 1000 -d gsm8k -m Qwen3-8B
CUDA_VISIBLE_DEVICES=3 python data_generation.py --device qz  --profile --max_gen 1000 -d alpaca -m Qwen3-8B
CUDA_VISIBLE_DEVICES=4 python data_generation.py --device qz  --profile --max_gen 1000 -d sum -m Qwen3-8B
CUDA_VISIBLE_DEVICES=5 python data_generation.py --device qz  --profile --max_gen 1000 -d vicuna-bench -m Qwen3-8B
CUDA_VISIBLE_DEVICES=6 python data_generation.py --device qz  --profile --max_gen 1000 -d math_infini -m Qwen3-8B