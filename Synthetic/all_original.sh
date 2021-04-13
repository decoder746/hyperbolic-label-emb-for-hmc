#!/bin/bash

python main.py --flat --change_prob 0.25 --exp_name o-flat-0.25
python main.py --flat --change_prob 0.75 --exp_name o-flat-0.75
python main.py --flat --change_prob 1 --exp_name o-flat-1
python main.py --joint --change_prob 0.25 --exp_name o-joint-0.25
python main.py --joint --change_prob 0.25 --exp_name o-joint-0.75
python main.py --joint --change_prob 1 --exp_name o-joint-1
