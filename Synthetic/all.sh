#!/bin/bash

python main_bilevel.py --joint --scale 100 --change_prob 0.25 --exp_name s-0.1-joint  &
python main_bilevel.py --joint --scale 100 --change_prob 0.5 --exp_name s-1-joint &
python main_bilevel.py --joint --scale 100 --change_prob  0.75 --exp_name s-10-joint &
python main_bilevel.py --joint --scale 100 --change_prob  1 --exp_name s-50-joint 
