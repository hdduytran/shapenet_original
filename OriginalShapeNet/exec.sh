#!/bin/bash
dataset=$1

python3 uea_original.py --dataset ${dataset} --gpu 1 --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json