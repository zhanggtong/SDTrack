PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=4,5 python tracking/train.py --script sdtrack --config SDTrack-ep150-full-256 --save_dir ./output --mode multiple --nproc_per_node 2
