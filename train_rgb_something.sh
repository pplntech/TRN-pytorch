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

	v19 ()
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

v09 on Ciplabthree (Sth-v2, h5)
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

v13 on DGX
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
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_0001_step_50100_clip_SORTING_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --lr_steps 50 100 --sorting --CustomPolicy



v21_bnFreeze_Adam_0_00001_step_100200_sorting_lstm on DGX (c)
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm
(V2_GPU3)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type h5 --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm



v21_bnFreeze_Adam_0_00001_step_100200_sorting_nosoftmax on DGX (compare with (b))
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_NOsoftmax_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --no_softmax_on_p
(V2_GPU1)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_NOsoftmax_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --no_softmax_on_p

(V2_GPU1 same network setting but different training setting.. such as  learning_step & learing_rate)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_0001_stepLR_default_clip_SORTING_NOsoftmax_v14/ --workers 20 --num_CNNs 1 \
--epochs 250 --file_type h5 --optimizer adam --lr 0.0001 --freezeBN --sorting --no_softmax_on_p



v21_bnFreeze_Adam_0_00001_step_100200_nosoftmax on kyungmin (compare with (a))
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --no_softmax_on_p



v21_bnFreeze_Adam_0_00001_step_100200_sorting_lstm on DGX (compare with (c))
python main.py something RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_NOsoftmax_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type jpg --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm --no_softmax_on_p
(V2_GPU4)
python main.py somethingv2 RGB --consensus_type MemNN --batch-size 30 --gpus 0 --root_path /raid/users/km/SthSth/ \
--key_dim 256 --value_dim 512 --query_dim 256 --query_update_method concat --hop_method parallel --num_segments 8 --hop 2 \
--result_path /raid/users/km/SthSth/Experiments/TRN/V2/v21_MemNNQueryNN_2hops_parallel_1CNN_bnFreeze_Adam_0_00001_step_100200_clip_SORTING_LSTM_NOsoftmax_v14/ --workers 20 --num_CNNs 1 --epochs 250 --file_type h5 --optimizer adam --lr 0.00001 --freezeBN --lr_steps 100 200 --sorting --how_to_get_query lstm --no_softmax_on_p

v22 SingleScaleTRN


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

 # create h5 files
 python utils/jpeg_to_h5_singleFile.py -t /raid/km/SthSth/20bn-something-something-v1-hdf5 -i /raid/km/SthSth/20bn-something-something-v1/ -e jpg -j 20

 # csv to tensorboard
 python utils/csv_to_tensorboard.py --result_path /hdd3/VideoDataset/Experiments/v09_MemNNQueryNN_1hop_1CNN/ -n 86017 -v 1 --train_parser [2860/2868] --val_parser [380/385