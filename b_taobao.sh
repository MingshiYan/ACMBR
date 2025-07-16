#!/bin/bash


lr=('0.001')
# lr=('0.001')
emb_size=(64)
reg_weight=('0.0001')
alphas=('0.0' '0.1' '0.3' '0.5' '0.7' '0.9' '1.0')

#behaviors=("['view', 'buy']" "['cart', 'buy']")
behaviors=("['view', 'cart', 'buy']")
ssl_tau=('0.05' )
ssl_weight=('0.3')
layers=2

dataset=('taobao')
device='cuda:1'
batch_size=1024
decay=('0')

data_loader='data_set'
model_name='model_lightGCN'
log_name='model_lightGCN'
gpu_no=1

for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
            for dec in ${decay[@]}
            do
            for alpha in ${alphas[@]}
            do
            for bhv in "${behaviors[@]}"
            do
            for tau in "${ssl_tau[@]}"
            do
            for ssl_w in "${ssl_weight[@]}"
            do
                echo 'start train: '$name
                `
                    python main.py \
                        --model_name $model_name \
                        --log_name $log_name \
                        --data_name $name \
                        --alpha $alpha \
                        --lr ${l} \
                        --layers $layers \
                        --behaviors "${bhv}" \
                        --ssl_tau "${tau}" \
                        --ssl_weight "${ssl_w}" \
                        --gpu_no $gpu_no \
                        --reg_weight ${reg} \
                        --embedding_size $emb \
                        --device $device \
                        --decay $dec \
                        --data_loader $data_loader \
                        --batch_size $batch_size 
                              
                `
                echo 'train end: '$name
            done
            done
            done
            done
            done
            done
        done
    done
done