#!/bin/bash

epochs=50


python face_learner.py -m pose_ht_4_unsupervised_final.pt -e $epochs -lr 0.05

# python face_learner.py -m pose_0_unsupervised_final.pt -e $epochs -lr 0.02

# python face_learner.py -m object_ht_9_unsupervised_final.pt -e $epochs -lr 0.05

# python face_learner.py -m pose_0.2_unsupervised_final.pt -e $epochs -lr 0.02

# python face_learner.py -m transform_ht_1_unsupervised_final.pt -e $epochs -lr 0.05

# python face_learner.py -m pose_0.4_unsupervised_final.pt -e $epochs -lr 0.02

# python face_learner.py -m pose_0.6_unsupervised_final.pt -e $epochs -lr 0.05

# python face_learner.py -m pose_0.6_unsupervised_final.pt -e $epochs -lr 0.02

# python face_learner.py -m pose_0.8_unsupervised_final.pt -e $epochs -lr 0.05

# python face_learner.py -m pose_0.8_unsupervised_final.pt -e $epochs -lr 0.02

# python face_learner.py -m pose_1_unsupervised_final.pt -e $epochs -lr 0.05

# python face_learner.py -m pose_1_unsupervised_final.pt -e $epochs -lr 0.02

