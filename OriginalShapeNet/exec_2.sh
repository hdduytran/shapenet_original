#!/bin/bash
dataset=$1

python3 uea_original.py --gpu 1 --dataset ${dataset} --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters_2.json