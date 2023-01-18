#!/bin/bash

epochs1=50
epochs2=40

python3 combined_learner.py -data toybox -d self -b 256 -w 8 -e1 $epochs1 -e2 $epochs2 -rt 0.0 -lr 0.005 -ht -m 0.4 -sv -sr -1 -sn pose_ht_5

python3 combined_learner.py -data toybox -d self -b 256 -w 8 -e1 $epochs1 -e2 $epochs2 -rt 0.0 -lr 0.01 -ht -m 0.4 -sv -sr -1 -sn pose_ht_6

python3 combined_learner.py -data toybox -d self -b 256 -w 8 -e1 $epochs1 -e2 $epochs2 -rt 0.0 -lr 0.02 -ht -m 0.4 -sv -sr -1 -sn pose_ht_7

python3 combined_learner.py -data toybox -d self -b 256 -w 8 -e1 $epochs1 -e2 $epochs2 -rt 0.0 -lr 0.05 -ht -m 0.4 -sv -sr -1 -sn pose_ht_8
