cd ../test
CUDA_VISIBLE_DEVICES=1 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 5120 -b 0 -e 40 > ../result/cal_sim/alpaca_skip_data_0.txt &
CUDA_VISIBLE_DEVICES=2 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 5120 -b 40 -e 80 > ../result/cal_sim/alpaca_skip_data_1.txt &
CUDA_VISIBLE_DEVICES=3 python search_mt_bench.py -d mt-bench -m QwQ -sim -sv -s  -n 5120 -b 0 -e 40 > ../result/cal_sim/mt-bench_skip_data_0.txt &
CUDA_VISIBLE_DEVICES=4 python search_mt_bench.py -d mt-bench -m QwQ -sim -sv -s  -n 5120 -b 40 -e 80 > ../result/cal_sim/mt-bench_skip_data_1.txt &
CUDA_VISIBLE_DEVICES=5 python search_mt_bench.py -d gsm8k -m QwQ -sim -sv -s  -n 5120 -b 0 -e 40 > ../result/cal_sim/gsm8k_skip_data_0.txt &
CUDA_VISIBLE_DEVICES=6 python search_mt_bench.py -d gsm8k -m QwQ -sim -sv -s  -n 5120 -b 40 -e 80 > ../result/cal_sim/gsm8k_skip_data_1.txt &
CUDA_VISIBLE_DEVICES=7 python search_mt_bench.py -d sum -m QwQ -sim -sv -s  -n 5120 -b 0 -e 40 > ../result/cal_sim/sum_skip_data_0.txt &
CUDA_VISIBLE_DEVICES=0 python search_mt_bench.py -d sum -m QwQ -sim -sv -s  -n 5120 -b 40 -e 80 > ../result/cal_sim/sum_skip_data_1.txt 
