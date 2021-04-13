#!/bin/bash

python main_bilevel.py --joint --scale 0.1 --exp_name s-0.1-joint &
python main_bilevel.py --joint --scale 1 --exp_name s-1-joint &
python main_bilevel.py --joint --scale 10 --exp_name s-10-joint &
python main_bilevel.py --joint --scale 50 --exp_name s-50-joint &
python main_bilevel.py --joint --scale 100 --exp_name s-100-joint 
