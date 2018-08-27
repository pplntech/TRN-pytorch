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
DGX : v03(128), v04(128), v05(80), v06(80, single hop), v07(80, 2 hops, 2 CNNs), v08(80, 2 hops, 1 CNN), v09(80, 2 jo)
Mine : v01, v02

V06 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 1 --result_path /raid/users/km/SthSth/Experiments/TRN/v06_simulatingTRN_byMemNN_num_seg8/ --workers 20

V07 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 2 --result_path /raid/users/km/SthSth/Experiments/TRN/v07_MemNNQueryNN_2hops/ --workers 20

V08 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 2 --result_path /raid/users/km/SthSth/Experiments/TRN/v08_MemNNQueryNN_2hops_1CNN/ --workers 20 --num_CNNs 1

v09 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 20 --gpus 0 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 2 --result_path /raid/users/km/SthSth/Experiments/TRN/v09_MemNNQueryNN_2hops_1CNN/ --workers 20 --num_CNNs 1

########################### TEST ########################### ()
python main.py something RGB --consensus_type MemNN --batch-size 20 --gpus ? --root_path ? \
--num_segments 8 --hop 2 --result_path ? --workers 20 --num_CNNs 2 --resume ? --evaluate --evaluation_epoch ?

python main.py something RGB --consensus_type MemNN --batch-size 20 --gpus 0 --root_path /hdd3/VideoDataset/ \
--num_segments 8 --hop 2 --result_path /hdd3/VideoDataset/Experiments/v07_MemNNQueryNN_2hops_2CNNs/ --workers 20 \
--num_CNNs 2 --resume /hdd3/VideoDataset/Experiments/v07_MemNNQueryNN_2hops_2CNNs/model/MemNN_something_RGB_BNInception_BNInception_MemNN_segment8_best.pth.tar --evaluate --evaluation_epoch 55


# create HTML files
python visualize_attention_score_HTML.py --img_root=/media/kyungmin/ThirdDisk1/VideoDataset/20bn-something-something-v1/ \
 --result_root=/media/kyungmin/ThirdDisk1/VideoDataset/Experiments/v07_MemNNQueryNN_2hops_2CNNs/ \
 --epoch=55 --category_path=/media/kyungmin/ThirdDisk1/VideoDataset/category_something-something-v1.txt --prefix={:05d}.jpg