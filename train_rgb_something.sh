python main.py something RGB --arch BNInception --num_segments 3 --consensus_type TRN --batch-size 48

# mine
python main.py something RGB --consensus_type MemNN --batch-size 32 --gpus 0 --root_path /hdd3/VideoDataset --num_segments 8 --hop 1 --result_path /hdd3/VideoDataset/Experiments/v02_temp

# dgx
# single hop
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 1 --result_path /raid/users/km/SthSth/Experiments/TRN/v05_MemNNQueryNN --workers 20
# multi hops
python main.py something RGB --consensus_type MemNN --batch-size 128 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 3 --result_path /raid/users/km/SthSth/Experiments/TRN/v03_temp --workers 20

# ciplabthree
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 39 --gpus 0 1 2 --root_path /hdd2/km/SthSth/ \
--num_segments 8 --hop 1 --result_path /hdd2/km/SthSth/Experiments/TRN/V2/v05_twoCNNs --workers 10


# Experiments (batch_size)
Ciplabthree : v05onV2(39)
DGX : v03(128), v04(128), v05(80), v06(80, single hop)
Mine : v01, v02

V06
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 1 --result_path /users/km/SthSth/Experiments/TRN/v06_MemNNQueryNN_2hops --workers 20