#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: Assassin 567
 # @Date: 2021-11-10 21:45:11
 # @LastEditors: Hello KG
 # @LastEditTime: 2021-11-24 10:41:44
### 

for l_w in 0 
do

for loss_error_weight in 0.1
do

for w_2 in 1e-4 
do

for DP_steps_set in 2 
do

for DP_num_edges_set in 15 
do

echo "GUP used : 0~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "DP_steps_set: $DP_steps_set .   DP_num_edges_set: $DP_num_edges_set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Search regularization-weight:  $w_2.  loss_margin  $l_w. loss_error_weight: $loss_error_weight"

CUDA_VISIBLE_DEVICES=0 python ./train.py --warm_start_time 48 --emb_dim 400 10 10 10 --batch_size 256 --lr 0.002 \
--dataset WIKI --epoch 2 --sampling 3 --device 0 --DP_steps $DP_steps_set --DP_num_edges 200 $DP_num_edges_set $DP_num_edges_set \
--max_attended_edges 30 --node_score_aggregation sum --ent_score_aggregation sum \
--ratio_update 0.75 --reg_weight $w_2 --loss_error_weight $loss_error_weight --loss_margin $l_w


done
done
done
done
done