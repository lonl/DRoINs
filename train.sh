CUDA_VISIBLE_DEVICES=0,1,2,3 python ./faster_rcnn/train_net.py\
 --gpu 0 --weights data/pretrain_model/Resnet50.npy --imdb VEDAI_1024_for_HPC \
 --iters 80000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --network Resnet50_train --restore 0

