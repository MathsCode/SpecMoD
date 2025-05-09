cd ../test
CUDA_VISIBLE_DEVICES=1 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 2048 -b 0 -e 1 > ../result/cal_sim/alpaca_skip_data_0.txt &
CUDA_VISIBLE_DEVICES=2 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 2048 -b 1 -e 2 > ../result/cal_sim/alpaca_skip_data_1.txt &
CUDA_VISIBLE_DEVICES=3 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 2048 -b 2 -e 3 > ../result/cal_sim/alpaca_skip_data_2.txt &
CUDA_VISIBLE_DEVICES=4 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 2048 -b 3 -e 4 > ../result/cal_sim/alpaca_skip_data_3.txt &
CUDA_VISIBLE_DEVICES=5 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 2048 -b 4 -e 5 > ../result/cal_sim/alpaca_skip_data_4.txt &
CUDA_VISIBLE_DEVICES=6 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 2048 -b 5 -e 6 > ../result/cal_sim/alpaca_skip_data_5.txt &
CUDA_VISIBLE_DEVICES=7 python search_mt_bench.py -d alpaca -m QwQ -sim -sv -s  -n 2048 -b 6 -e 7 > ../result/cal_sim/alpaca_skip_data_6.txt 
