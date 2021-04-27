#!/bin/bash

python main_bilevel.py --joint --scale 100 --change_prob 0.25 --exp_name v1-0.25-joint  &
python main_bilevel.py --joint --scale 100 --change_prob 0.5 --exp_name v1-0.5-joint &
python main_bilevel.py --joint --scale 100 --change_prob  0.75 --exp_name v1-0.75-joint &
python main_bilevel.py --joint --scale 100 --change_prob  1 --exp_name v1-1-joint 
