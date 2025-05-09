cd ../test
CUDA_VISIBLE_DEVICES=0 python search_mt_bench.py -m QwQ -s -n 16 -b 0 -e 1 -SL 64 > ../result/search/test_0.txt &
CUDA_VISIBLE_DEVICES=1 python search_mt_bench.py -m QwQ -s -n 16 -b 1 -e 2 -SL 64 > ../result/search/test_1.txt &
CUDA_VISIBLE_DEVICES=2 python search_mt_bench.py -m QwQ -s -n 16 -b 2 -e 3 -SL 64 > ../result/search/test_2.txt &
CUDA_VISIBLE_DEVICES=3 python search_mt_bench.py -m QwQ -s -n 16 -b 3 -e 4 -SL 64 > ../result/search/test_3.txt &
CUDA_VISIBLE_DEVICES=4 python search_mt_bench.py -m QwQ -s -n 16 -b 4 -e 5 -SL 64 > ../result/search/test_4.txt &
CUDA_VISIBLE_DEVICES=5 python search_mt_bench.py -m QwQ -s -n 16 -b 5 -e 6 -SL 64 > ../result/search/test_5.txt &
CUDA_VISIBLE_DEVICES=6 python search_mt_bench.py -m QwQ -s -n 16 -b 6 -e 7 -SL 64 > ../result/search/test_6.txt &
CUDA_VISIBLE_DEVICES=7 python search_mt_bench.py -m QwQ -s -n 16 -b 7 -e 8 -SL 64 > ../result/search/test_7.txt 
