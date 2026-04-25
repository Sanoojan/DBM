

# first extract optic flow features (Only once)
CUDA_VISIBLE_DEVICES=1 python data_preprocess_protocol.py 

# then train the model
# Backbone r3d18, r2plus1d_18, mc3_18, resnet18, vit_b_16, resnet50
CUDA_VISIBLE_DEVICES=1 python Strain1.py --model VideoModel --backbone r2plus1d_18  --batch_size 8 > Logs/BENCHMARK_r2plus1d_18_MIL_ranking_std.log 2>&1 &

# Then aggregate the results for easy access
# Please change the LOG dir inside the aggregator.py to the dir where you save the logs
python aggregator.py