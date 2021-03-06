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
Ciplabthree
	v00onV2(256, MultiScaleTRN)
	v05onV2(39)
	v09onV2(81)
DGX
	v03 (128, MemNN, 3 hops, 1 CNN, out, FAILED)
	v04 (128, TRN, num_seg 7, 31.748) --consensus_type TRN, num_segments 7
	v05 (80, MemNN, 1 hop, 2 CNNs, out+query, 20.040)
	v06 (80, single hop, JUST for testing, SIMULATING v04 by memNN)
	v07 (80, 2 hops, 2 CNNs, out+query)
	v08 (80, 2 hops, 1 CNN, out+query)
	--------------vv valid vv-------------- (1 CNN)
	v09 (30, 1 hop) && v19 (30, 1 hop) # same with v09 except for optimizer(Adam)
	v10 (30, 2 hops, iterative, concat) && v20 (30, 2 hops, iterative, concat, 0.001, FAILED) && so and on # same with v10 except for optimizer(Adam)
	v11 (30, 2 hops, iterative, sum, 1 CNN)
	# v12 (30, 1 hop, addWhenQueryUpdating, value256, 1 CNN) # compare with v9! # JUST for testing
	# v13 (30, 1 hop, 1 CNN, no_clip_gradients) # same as v9 except for existence of clip_gradients # JUST for testing
	v14 (30, 2 hops, parallel) && v21 (30, 2 hops, parallel, 0.001, FAILED) && so and on # same with v14 except for optimizer(Adam) [# compare with v09,10]
	v15 (30, 3 hops, parallel) # compare with v09,10
	v16 (30, 3 hops, iterative, concat) # compare with v09,10
	# v17 (30, 3 hops, iterative, sum, 1 CNN) # compare with v09,11, Failed
	v18 (30, 3 hops, iterative, sum) # compare with v09,11, multiple query embedding, RUNNING ! 

	v19 (1 hop)
	v20 (iterative, concat)
	v21 (parallel)
	--------------vv valid vv--------------
	
	
	
Mine
	v01 (TRN, num_seg 2, 21.863)
	v02 (MemNN, 1 hop, 1CNN, out+query_emb, 19.997)


Ciplabthree
v00 on Ciplabthree (Sth-v2)
python main.py somethingv2 RGB --consensus_type TRNmultiscale --batch-size 81 --gpus 0 1 2 --root_path /ssd/km/SthSth/ \
--img_feature_dim 256 --num_segments 8 --hop 1 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v00_MulitScaleTRN --workers 20 --num_CNNs 1 --epochs 250 --file_type h5

v05 on Ciplabthree (Sth-v2)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 39 --gpus 0 1 2 --root_path /hdd2/km/SthSth/ \
--num_segments 8 --hop 1 --result_path /hdd2/km/SthSth/Experiments/TRN/V2/v05_twoCNNs --workers 20

v09 on Ciplabthree (Sth-v2)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 81 --gpus 0 1 2 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --num_segments 8 --hop 1 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v09_MemNNQueryNN_1hop_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type h5

v22 on Ciplabthree (Sth-v2)
python main.py somethingv2 RGB --consensus_type TRN --batch-size 27 --gpus 0 --root_path /ssd/km/SthSth/ \
--img_feature_dim 256 --num_segments 2 --hop 1 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v22_SingleScaleTRN_frame2 --workers 20 --num_CNNs 1 --epochs 250 --file_type h5

DGX
v06 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 1 --result_path /raid/users/km/SthSth/Experiments/TRN/v06_simulatingTRN_byMemNN_num_seg8/ --workers 20

v07 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 2 --result_path /raid/users/km/SthSth/Experiments/TRN/v07_MemNNQueryNN_2hops/ --workers 20

v08 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 80 --gpus 0 1 2 3 --root_path /raid/users/km/SthSth/ \
--num_segments 8 --hop 2 --result_path /raid/users/km/SthSth/Experiments/TRN/v08_MemNNQueryNN_2hops_1CNN/ --workers 20 --num_CNNs 1

v09 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --num_segments 8 --hop 1 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v09_MemNNQueryNN_1hop_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --resume ? 

v10 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v10_MemNNQueryNN_2hops_concat_iter_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --resume ? 

v11 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 1024 --query_dim 256 --query_update_method sum --hop_method iterative --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v11_MemNNQueryNN_2hops_sum_iter_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --resume ?

v12 on DGX # TEMP version, addWhenQueryUpdating
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 256(must be same with query_dim) --query_dim 256 --num_segments 8 --hop 1 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v12_MemNNQueryNN_1hop_1CNN_queryupdateTEMP/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --resume ?

v13 on DGX # just for TESTING
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --num_segments 8 --hop 1 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v13_MemNNQueryNN_1hop_1CNN_noClipGradient/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --no_clip

v14 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v14_MemNNQueryNN_2hops_parallel_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg

v15 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v15_MemNNQueryNN_3hops_parallel_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg

v16 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v16_MemNNQueryNN_3hops_concat_iter_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg

v17/v18 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 1024 --query_dim 256 --query_update_method sum --hop_method iterative --num_segments 8 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v17_MemNNQueryNN_3hops_sum_iter_1CNN/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg

v19 on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --num_segments 8 --hop 1 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v19_MemNNQueryNN_1hop_1CNN_v09/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.001
v19_sgd on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --num_segments 8 --hop 1 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v19_MemNNQueryNN_1hop_1CNN_v09_sgd/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer sgd --lr 0.001
v19 with CustomPolicy
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 1 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v19_MemNNQueryNN_1hop_1CNN_bnFreeze_Adam_0_0001_stepLR_default_YesClip_CustomPolicy/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --CustomPolicy


v20 (iter, concat)
# v20 on DGX (filed, too high lr)
# python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
# --key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 2 \
# --result_path /raid/users/km/SthSth/Experiments/TRN/v20_MemNNQueryNN_2hops_concat_iter_1CNN_v10/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.001

# v20_bnFreeze_Adam_0_00001_step_100200_noclip on DGX (it runs but slow, stopped in the middle)
# python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
# --key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 2 \
# --result_path /raid/users/km/SthSth/Experiments/TRN/v20_MemNNQueryNN_2hops_concat_iter_1CNN_bnFreeze_Adam_0_00001_step_100200_noclip_v10/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --no_clip

v20_bnFreeze_Adam_0_00001_step_100200 on DGX (finetuned)
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v20_MemNNQueryNN_2hops_concat_iter_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_v10/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200

v20_bnFreeze_Adam_0_0001_step_50100_freezeBackbone_Noclip_v10
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v20_MemNNQueryNN_2hops_concat_iter_1CNN_bnFreeze_Adam_0_0001_step_50100_freezeBackbone_Noclip_v10/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN--freezeBackbone --lr_steps 50 100 --no_clip

# v20_bnFreeze_Adam_0_001_step_60120_freezeBackbone_clip on DGX  (failed, too high lr)
# python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
# --key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 2 \
# --result_path /raid/users/km/SthSth/Experiments/TRN/v20_MemNNQueryNN_2hops_concat_iter_1CNN_bnFreeze_Adam_0_001_step_60120_freezeBackbone_clip_v10/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.001 --freezeBN --freezeBackbone --lr_steps 60 120

v20_freezeBackbone on my computer (kyungmin)
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /hdd3/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8 --hop 2 \
--result_path /hdd3/VideoDataset/Experiments/tmp_iter_concat_freezeBackbone --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.001 --freezeBN --freezeBackbone --lr_steps 100 200


v21 (parallel)
# v21 on DGX (failed, too high lr)
# python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
# --key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
# --result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.001

# v21_bnFreeze_Adam_0_00001_step_100200_noclip on DGX (it runs but slow, stopped in the middle)
# python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
# --key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
# --result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_noclip_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --no_clip

v21_bnFreeze_Adam_0_00001_step_100200 on DGX (FINETUNED training setting on my algorithm and sht-sth dataset) (a)
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200

# v21_bnFreeze_Adam_0_001_step_60120_freezeBackbone_clip on DGX (failed, too high lr)
# python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
# --key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
# --result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_001_step_60120_freezeBackbone_clip_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.001 --freezeBN --freezeBackbone --lr_steps 60 120

v21_bnFreeze_Adam_0_00001_step_freezeBackbone_noclip on DGX
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_0001_step_50100_freezeBackbone_Noclip_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.0001 --freezeBN --freezeBackbone --lr_steps 50 100 --no_clip



v21_bnFreeze_Adam_0_00001_step_100200_sorting on DGX (b)
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting
(V2_GPU0)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting
(V2_GPU0 same network setting but different training setting.. such as  learning_step & learing_rate & custompolicy)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_CustomPolicy_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --CustomPolicy



v21_bnFreeze_Adam_0_00001_step_100200_sorting_lstm on DGX (c)
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm
(V2_GPU3)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm
(V2_GPU3 hop3 same network setting but different training setting.. such as  learning_step & learing_rate & custompolicy)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_LSTM_CustomPolicy_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --CustomPolicy



v21_bnFreeze_Adam_0_00001_step_100200_sorting_nosoftmax on DGX (compare with (b))
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_NOsoftmax_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --no_softmax_on_p
(V2_GPU1)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_NOsoftmax_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --no_softmax_on_p
(V2_GPU1 same network setting but different training setting.. such as  learning_step & learing_rate & custompolicy)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_CustomPolicy_NOsoftmax_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --no_softmax_on_p --CustomPolicy



v21_bnFreeze_Adam_0_00001_step_100200_nosoftmax on kyungmin (compare with (a))
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --no_softmax_on_p



v21_bnFreeze_Adam_0_00001_step_100200_sorting_lstm on DGX (compare with (c))
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_NOsoftmax_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm --no_softmax_on_p
(V2_GPU4)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_NOsoftmax_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm --no_softmax_on_p
(V2_GPU4 hop3 same network setting but different training setting.. such as  learning_step & learing_rate & custompolicy)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy
(V2_GPU3 hop3 same network setting but different training setting.. such as  learning_step & learing_rate & custompolicy && num_segments 16)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 18 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_numseg16_v14/ \
--workers 20 --num_CNNs 1 --epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy



v22 SingleScaleTRN on V2
TRN-fr2
TRN-fr3
TRN-fr7
python main.py somethingv2 RGB --consensus_type TRN --batch-size 27 --gpus 0 \
--root_path /ssd/km/SthSth/ --img_feature_dim 256 --num_segments 7 --hop 1 --result_path /hdd2/km/SthSth/Experiments/TRN/V2/v22_SingleScaleTRN_frame7 \
--workers 20 --num_CNNs 1 --epochs 250 --file_type h5
TRN-fr16 (ciplabeight)
CUDA_VISIBLE_DEVICES=1,2,3 python main.py something RGB --consensus_type TRN --batch-size 36 --gpus 0 1 2 --root_path /ssd1/users/km/VideoDataset/ \
--img_feature_dim 256 --num_segments 16 --hop 1 --result_path /hdd1/users/km/SthSth/Experiments/TRN/V1/TMP \
--workers 20 --num_CNNs 1 --epochs 350 --file_type h5


v23 only_LSTM (on my computer) BNInception frame_num 8
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v23_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_ONLYLSTM --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --only_query

v23 only_LSTM (on my computer) BNInception frame_num 16
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 14 --gpus 0 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16 --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v23_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_ONLYLSTM_numseg16 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --only_query

v23 only_LSTM (on my computer) ResNet50 frame_num 8
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 14 --gpus 0 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v23_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_ONLYLSTM_Res50_num8 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN_Eval --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --only_query \
--arch resnet50 --channel 2048 --freezeBN_Require_Grad_True --npb \
--resume /hdd3/VideoDataset/Experiments/V2/v23_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_ONLYLSTM_Res50_num8/model/MemNN_somethingv2_RGB_resnet50_MemNN_segment8_key256_value512_query256_queryUpdatebyconcat_NoSoftmaxTrue_hopMethodparallel_checkpoint.pth.tar

v23 only_LSTM (on dgx GPU3) ResNet50 frame_num 16
CUDA_VISIBLE_DEVICES=2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 9 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v23_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_ResNet50_ONLYLSTM_numseg16 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --only_query \
--arch resnet50 --channel 2048 --freezeBN_Grad --npb

v24 hop3 ResNet50 CC (ciplabthree GPU2)
CUDA_VISIBLE_DEVICES=2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 14 --gpus 0 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v24_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_LSTM_NOsoftmax_CustomPolicy_CC_ResNet50/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb --lr_steps 55 70 \
--resume /hdd2/km/SthSth/Experiments/TRN/V2/v24_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_LSTM_NOsoftmax_CustomPolicy_CC_ResNet50/model/MemNN_somethingv2_RGB_resnet50_MemNN_segment8_key256_value512_query256_queryUpdatebyconcat_NoSoftmaxTrue_hopMethodparallel_best.pth.tar

v24 hop3 ResNet50 NoCC NoAdditionalLoss (ciplabthree GPU0, 181008)
CUDA_VISIBLE_DEVICES=0 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 14 --gpus 0 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v24_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_LSTM_NOsoftmax_CustomPolicy_NoCC_ResNet50_LrSteps3570/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --arch resnet50 --channel 2048 --freezeBN_Grad --npb --lr_steps 35 70

v24 hop3 ResNet50 CC num_seg 16 (ciplabthree GPU0,1)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 12 --gpus 0 1 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v24_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_LSTM_NOsoftmax_CustomPolicy_CC_ResNet50_numseg16/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb

v25 hop3 ResNet50 CC AdditionalLoss iterative num_segments_8 (ciplabthree GPU0)
CUDA_VISIBLE_DEVICES=0 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 12 --gpus 0 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v25_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_iterative_numseg8/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb --AdditionalLoss --lr_steps 35 70 \
--resume /hdd2/km/SthSth/Experiments/TRN/V2/v25_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_iterative_numseg8/model/MemNN_somethingv2_RGB_resnet50_MemNN_segment8_key256_value512_query256_queryUpdatebyconcat_NoSoftmaxTrue_hopMethoditerative_best.pth.tar

v25 hop3 ResNet50 CC AdditionalLoss iterative num_segments_16 (ciplabthree GPU1)
CUDA_VISIBLE_DEVICES=1 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 6 --gpus 0 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v25_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_iterative_numseg16/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb --AdditionalLoss --lr_steps 35 70 \
--resume /hdd2/km/SthSth/Experiments/TRN/V2/v25_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_iterative_numseg16/model/MemNN_somethingv2_RGB_resnet50_MemNN_segment16_key256_value512_query256_queryUpdatebyconcat_NoSoftmaxTrue_hopMethoditerative_best.pth.tar

v26 hop3 ResNet50 CC AdditionalLoss parallel num_segments_8 (dgx GPU0)
CUDA_VISIBLE_DEVICES=0 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 18 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v26_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_parallel_numseg8 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb --AdditionalLoss --lr_steps 35 70 \
--resume /raid/users/km/SthSth/Experiments/TRN/V2/v26_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_parallel_numseg8/model/MemNN_somethingv2_RGB_resnet50_MemNN_segment8_key256_value512_query256_queryUpdatebyconcat_NoSoftmaxTrue_hopMethodparallel_best.pth.tar

v26 hop3 ResNet50 CC AdditionalLoss parallel num_segments_16 (dgx GPU1)
CUDA_VISIBLE_DEVICES=1 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 9 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v26_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_parallel_numseg16 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb --AdditionalLoss --lr_steps 35 70 \
--resume /raid/users/km/SthSth/Experiments/TRN/V2/v26_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_parallel_numseg16/model/MemNN_somethingv2_RGB_resnet50_MemNN_segment16_key256_value512_query256_queryUpdatebyconcat_NoSoftmaxTrue_hopMethodparallel_best.pth.tar

v27 (on my computer) 2D
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 14 --gpus 0 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8  --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/TMP --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --how_to_get_query mean --CustomPolicy \
--CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb --AdditionalLoss --AdditionalLoss_MLP --memory_dim 2

v28 (dgx GPU1) ResNet18 h5_320 DataAugmentation_Rotation_ColorJittering MLP_on_MultiStageLoss (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=1 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 50 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8  --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v28_MemNNQueryNN_3hops_NOsoftmax_ResNet18_MultiStageLossMLP_iter_numseg8/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet18 --channel 512 --freezeBN_Eval --npb  --lr_steps 30 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320

v28 (dgx GPU1) ResNet34 h5_320 DataAugmentation_Rotation_ColorJittering MLP_on_MultiStageLoss (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=1 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 42 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 8  --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v28_MemNNQueryNN_3hops_NOsoftmax_ResNet34_MultiStageLossMLP_iter_numseg8/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 35 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320

v28 (dgx GPU3) ResNet18 h5_320 DataAugmentation_Rotation_ColorJittering MLP_on_MultiStageLoss (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 25 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v28_MemNNQueryNN_3hops_NOsoftmax_ResNet18_MultiStageLossMLP_iter_numseg16/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet18 --channel 512 --freezeBN_Eval --npb  --lr_steps 30 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320

v29 (ciplabthree GPU0) ResNet18 h5_320 DataAugmentation_Rotation_ColorJittering MLP_on_MultiStageLoss (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=0 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 42 --gpus 0 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v29_MemNNQueryNN_3hops_NOsoftmax_ResNet18_MultiStageLossMLP_parallel_numseg8_BNGradTrue/ --workers 5 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet18 --channel 512 --freezeBN_Eval --freezeBN_Require_Grad_True --npb  --lr_steps 30 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --image_resolution 320

v29 (ciplabthree GPU1) ResNet18 h5_320 DataAugmentation_Rotation_ColorJittering MLP_on_MultiStageLoss (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=1 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 39 --gpus 0 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v29_MemNNQueryNN_3hops_NOsoftmax_ResNet18_MultiStageLossMLP_parallel_numseg8_BNGradFalse/ --workers 5 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet18 --channel 512 --freezeBN_Eval --npb  --lr_steps 30 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --image_resolution 320

v29 (ciplabthree GPU0,1) ResNet34 h5_320 DataAugmentation_Rotation_ColorJittering MLP_on_MultiStageLoss (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=0,1 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 66 --gpus 0 1 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v29_MemNNQueryNN_3hops_NOsoftmax_ResNet34_MultiStageLossMLP_parallel_numseg8_BNGradFalse/ --workers 5 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 30 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --image_resolution 320

v29 (ciplabthree GPU2) ResNet18 h5_320 DataAugmentation_Rotation_ColorJittering MLP_on_MultiStageLoss (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 21 --gpus 0 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v29_MemNNQueryNN_3hops_NOsoftmax_ResNet18_MultiStageLossMLP_parallel_numseg16/ --workers 5 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet18 --channel 512 --freezeBN_Eval --npb  --lr_steps 30 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --image_resolution 320

v30 (on my computer) ResNet34 frame_num16 parallel softmax sorting (h5 : 256)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 1 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16  --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v30_ResNet34_16frs_Parallel_Softmax_Sorting --workers 8 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 35 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 256

v30 (dgx GPU1,3) ResNet34 frame_num16 parallel Nosoftmax Nosorting (default : LSTM for query, CC, MultiStageLoss) (h5 : 256)
CUDA_VISIBLE_DEVICES=1,2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 46 --gpus 0 1 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16  --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v30_ResNet34_16frs_Parallel_NoSoftmax_NoSorting/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 35 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 256

v30 (ciplabthree GPU0,1,2) ResNet34 frame_num16 iter Nosoftmax Nosorting (default : LSTM for query, CC, MultiStageLoss) (h5 : 256)
CUDA_VISIBLE_DEVICES=0,1,2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 48 --gpus 0 1 2 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v30_ResNet34_16frs_Iterative_NoSoftmax_NoSorting/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 35 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 256

v31 (dgx GPU1,3) ResNet34 frame_num16 iter Nosoftmax Nosorting num_objects 2 (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=1,2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 46 --gpus 0 1 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v31_ResNet34_16frs_Iter_NoSoftmax_NoSorting_numobjects2/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 35 50 70 --MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --how_many_objects 2

v31 (ciplabthree GPU0,1,2) ResNet34 frame_num16 iter Nosoftmax Nosorting num_objects 2 2D extension (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=0,1,2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 48 --gpus 0 1 2 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v31_ResNet34_16frs_Iter_NoSoftmax_NoSorting_numobjects2_memorydim2/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 40 50 70 \
--MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --how_many_objects 2 --memory_dim 2

v31 (on my computer) ResNet34 frame_num16 iter Nosoftmax Nosorting Each_Embedding (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 1 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v31_ResNet34_16frs_Iter_NoSoftmax_NoSorting_Each_Embedding/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet34 --channel 512 --freezeBN_Eval --npb  --lr_steps 35 50 70 \
--MultiStageLoss --MultiStageLoss_MLP --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --Each_Embedding

v32 (on my computer) Curriculum Learning ResNet50 frame_num16 iter softmax Nosorting (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 12 --gpus 0 1 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v32_ResNet50_16frs_Iterative_YesSoftmax_NoSorting/ --workers 8 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm \
--CustomPolicy --CC --arch resnet50 --channel 2048 --lr_steps 15 30 40 50 --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --Curriculum --Curriculum_dim 512 --npb --freezeBN_Eval

v32 (on my computer) Curriculum Learning ResNet50 frame_num16 iter Nosoftmax Nosorting (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 12 --gpus 0 1 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v32_ResNet50_16frs_Iterative_NoSoftmax_NoSorting/ --workers 8 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet50 --channel 2048 --lr_steps 15 30 40 50 --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --Curriculum --Curriculum_dim 512 --npb --freezeBN_Eval

############# TEST #############
NO CURRICULUM
v32 (on my computer) ResNet50 frame_num16 iter Nosoftmax Nosorting (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
[refer to v26 hop3 ResNet50 CC AdditionalLoss parallel num_segments_16 (dgx GPU1)]
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 9 --gpus 0 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16 --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v32_TEST_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_parallel_numseg16 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --CC \
--arch resnet50 --channel 2048 --freezeBN_Grad --npb --AdditionalLoss --lr_steps 35 70 \

NO CURRICULUM
v32 (on my computer) ResNet50 frame_num16 iter Nosoftmax Nosorting (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
[refer to v23 only_LSTM ResNet50 frame_num 16 (on dgx GPU3)]
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 9 --gpus 0 --root_path /ssd2/VideoDataset/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16 --hop 3 \
--result_path /hdd3/VideoDataset/Experiments/V2/v32_TEST_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_default_clip_SORTING_LSTM_NOsoftmax_CustomPolicy_ResNet50_ONLYLSTM_numseg16 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --sorting --how_to_get_query lstm --no_softmax_on_p --CustomPolicy --only_query \
--arch resnet50 --channel 2048 --freezeBN_Grad --npb
############# TEST #############

v32 (dgx GPU1,3) Curriculum Learning ResNet50 frame_num16 iter Nosoftmax Nosorting (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=1,2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 20 --gpus 0 1 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v32_ResNet50_16frs_Iterative_NoSoftmax_NoSorting/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p  \
--CustomPolicy --CC --arch resnet50 --channel 2048 --lr_steps 15 30 40 50 --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --Curriculum --Curriculum_dim 512 --npb

############# TEST #############
CUDA_VISIBLE_DEVICES=1,2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 20 --gpus 0 1 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16  --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v32_ResNet50_16frs_Iterative_NoSoftmax_NoSorting/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p  \
--CustomPolicy --CC --arch resnet50 --channel 2048 --lr_steps 15 30 40 50 --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --npb

v26 hop3 ResNet50 CC AdditionalLoss parallel num_segments_16 (dgx GPU1)
CUDA_VISIBLE_DEVICES=1 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 9 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 16 --hop 3 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v26_MemNNQueryNN_3hops_LSTM_NOsoftmax_CC_ResNet50_AdditionalLoss_parallel_numseg16 --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --how_to_get_query lstm --no_softmax_on_p \
--CustomPolicy --CC --arch resnet50 --channel 2048 --freezeBN_Grad --npb --AdditionalLoss --lr_steps 35 70 \
############# TEST #############

v32 (ciplabthree GPU0,1,2) Curriculum Learning ResNet50 frame_num16 iter Nosoftmax Nosorting Each_Embedding (default : LSTM for query, CC, MultiStageLoss) (h5 : 320)
CUDA_VISIBLE_DEVICES=0,1,2 python main.py somethingv2 RGB --consensus_type MemNN --batch-size 20 --gpus 0 1 2 --root_path /ssd/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method iterative --num_segments 16  --hop 3 \
--result_path /hdd2/km/SthSth/Experiments/TRN/V2/v32_ResNet50_16frs_Iterative_NoSoftmax_NoSorting_EachEmbedding/ --workers 20 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --how_to_get_query lstm --no_softmax_on_p  \
--CustomPolicy --CC --arch resnet50 --channel 2048 --lr_steps 15 30 40 50 --MoreAug_Rotation --MoreAug_ColorJitter --image_resolution 320 --Curriculum --Curriculum_dim 512 --Each_Embedding --npb


########################### TEST ########################### ()
python main.py something RGB --consensus_type MemNN --batch-size 20 --gpus ? --root_path ? \
--num_segments 8 --hop 2 --result_path ? --workers 20 --num_CNNs 2 --resume ? --evaluate --evaluation_epoch ?

python main.py something RGB --consensus_type MemNN --batch-size 20 --gpus 0 --root_path /hdd3/VideoDataset/ \
--num_segments 8 --hop 2 --result_path /hdd3/VideoDataset/Experiments/v07_MemNNQueryNN_2hops_2CNNs/ --workers 20 \
--num_CNNs 2 --resume /hdd3/VideoDataset/Experiments/v07_MemNNQueryNN_2hops_2CNNs/model/MemNN_something_RGB_BNInception_BNInception_MemNN_segment8_best.pth.tar --evaluate --evaluation_epoch 55


# create HTML files
python ./utils/visualize_attention_score_HTML.py --img_root=/media/kyungmin/ThirdDisk1/VideoDataset/20bn-something-something-v1/ \
 --result_root=/media/kyungmin/ThirdDisk1/VideoDataset/Experiments/v07_MemNNQueryNN_2hops_2CNNs/ \
 --epoch=55 --category_path=/media/kyungmin/ThirdDisk1/VideoDataset/category_something-something-v1.txt --prefix={:05d}.jpg
 python ./utils/visualize_attention_score_HTML.py --img_root=/media/kyungmin/ThirdDisk1/VideoDataset/20bn-something-something-v2/ \
 --result_root=/media/kyungmin/ThirdDisk1/VideoDataset/Experiments/V2/v21_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_LSTM_CustomPolicy_v14/ \
 --epoch=200 --category_path=/media/kyungmin/ThirdDisk1/VideoDataset/category_something-something-v2.txt  --prefix={:06d}.jpg

 # create h5 files
 python utils/jpeg_to_h5_singleFile.py -t /raid/km/SthSth/20bn-something-something-v1-hdf5 -i /raid/km/SthSth/20bn-something-something-v1/ -e jpg -j 20

 # json to accuracy
 python ./utils/json_to_per_class_acc.py --img_root=/media/kyungmin/ThirdDisk1/VideoDataset/20bn-something-something-v2 \
 --result_root=/media/kyungmin/ThirdDisk1/VideoDataset/Experiments/V2/v21_MemNNQueryNN_3hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_LSTM_CustomPolicy_v14/ \
 --epoch=200 --category_path=/media/kyungmin/ThirdDisk1/VideoDataset/category_something-something-v2.txt --confusion_topk=20

 # csv to tensorboard
 python utils/csv_to_tensorboard.py --result_path /hdd3/VideoDataset/Experiments/v09_MemNNQueryNN_1hop_1CNN/ -n 86017 -v 1 --train_parser [2860/2868] --val_parser [380/385]

 # nvidia-docker (ciplabeight)
docker pull yonsei-cip2.synology.me:5443/km/pytorch_angusism:latest
NV_GPU=0,1,2,3 nvidia-docker run -ti --name KM_STHSTH_gpu0123 --ipc=host \
-v /home/ciplab/users:/workspace -v /media/volume1:/hdd1 -v /media/volume2:/hdd2 -v /media/ssd1:/ssd1 yonsei-cip2.synology.me:5443/km/pytorch_angusism:latest

nvidia-docker (my computer)
NV_GPU=0,1 nvidia-docker run -ti --name KM_pytorch_GPU01 --ipc=host \
-v ~/:/workspace -v /hdd2:/hdd2 -v /media/kyungmin/8EACE8DCACE8BFB71:/hdd1 -v /media/kyungmin/ssd2:/ssd2 -v /media/kyungmin/ThirdDisk1:/hdd3 -v /mnt/NAS/:/NAS \
 heyday097/pytorch:latest

# extract jpeg v2
python utils/extract_frames.py --video_root /raid/km/SthSth/20bn-something-something-v2-vids/ --frame_root /raid/km/SthSth/20bn-something-something-v2-frames_320_jpegQual5 --num_threads 20 --resolution 256 --quality 7