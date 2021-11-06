#!/bin/bash

for l_w in 0  
do

for loss_error_weight in 0.01 
do

for w_2 in 1e-3 
do

for DP_steps_set in 2 
do

for DP_num_edges_set in 30 
do


echo "GUP used : 1"
echo "DP_steps_set: $DP_steps_set .   DP_num_edges_set: [200,$DP_num_edges_set,$DP_num_edges_set]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Search regularization-weight:  $w_2.  loss_margin  $l_w. loss_error_weight: $loss_error_weight"

CUDA_VISIBLE_DEVICES=3 python ./train.py --warm_start_time 48 --emb_dim 400 10 10 10 --batch_size 256 --lr 0.002 \
--dataset ICEWS0515_forecasting --epoch 4 --sampling 3 --device 0 --DP_steps $DP_steps_set --DP_num_edges 200 $DP_num_edges_set $DP_num_edges_set \
--max_attended_edges 40 --node_score_aggregation sum --ent_score_aggregation sum \
--ratio_update 0 --reg_weight $w_2 --loss_error_weight $loss_error_weight --loss_margin $l_w

done
done
done
done
done