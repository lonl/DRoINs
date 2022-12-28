#!/bin/bash

python ./faster_rcnn/train_net.py\
 --gpu 0 --weights data/pretrain_model/Resnet50.npy --imdb VEDAI_1024_for_HPC \
 --iters 1000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --network Resnet50_train --restore 0

# generate an image
if [ ! -f experiments/profiling/gprof2dot.py ]; then 
	echo "Downloading ... "
	wget https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py -O experiments/profiling/gprof2dot.py
fi
python experiments/profiling/gprof2dot.py -f pstats experiments/profiling/profile.out | dot -Tpng -o experiments/profiling/profile.png
