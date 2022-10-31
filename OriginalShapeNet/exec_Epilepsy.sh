#!/bin/bash
python3 uea_original.py --dataset EthanolConcentration --path ../../dataset --save_path ../../shapenet_results --cuda --hyper default_parameters.json --ratio 0.5
python3 uea_original.py --dataset EthanolConcentration --path ../../dataset --save_path ../../shapenet_results --cuda --hyper default_parameters.json --ratio 0.7
python3 uea_original.py --dataset EthanolConcentration --path ../../dataset --save_path ../../shapenet_results --cuda --hyper default_parameters.json --ratio 0.1
python3 uea_original.py --dataset EthanolConcentration --path ../../dataset --save_path ../../shapenet_results --cuda --hyper default_parameters.json --ratio 0.3
python3 uea_original.py --dataset EthanolConcentration --path ../../dataset --save_path ../../shapenet_results --cuda --hyper default_parameters.json --ratio 1
