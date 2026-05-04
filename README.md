# TrustworthyML_2026
This project implements a Membership Inference Attack (MIA) on a pretrained ResNet-18 image classification model. The goal is to determine whether a given image was part of the model’s training dataset by analyzing the model’s output behavior.  

#### Given a set of samples without membership labels, the system assigns a confidence score between 0 and 1, indicating the likelihood that each sample was used during training. Higher scores correspond to higher membership probability.

#### The attack is evaluated using TPR@5% FPR, focusing on performance under strict false-positive constraints.
