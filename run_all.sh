# ./run_experiments.py -e exp/example.txt --experiments_per_gpu 2 --num_gpu 2

./analysis/feature_test.py
./run_experiments.py -e exp/example.txt --experiments_per_gpu 1 --num_gpu 1
./analysis/itsc25.py checkpoints