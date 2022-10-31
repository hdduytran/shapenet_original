#!/bin/bash
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.1
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.2
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.3
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.4
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.5
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.6
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.7
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.8
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 0.9
python3 uea_original.py --dataset Heartbeat --path ../../dataset --save_path ./shapenet_results --cuda --hyper default_parameters.json --ratio 1
