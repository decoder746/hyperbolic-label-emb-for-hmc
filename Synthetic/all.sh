#!/bin/bash

python main_bilevel.py --joint --scale 10 --q 0.1 --exp_name q-0.1-joint &
python main_bilevel.py --joint --scale 10 --q 1  --exp_name q-1-joint &
python main_bilevel.py --joint --scale 10 --q 10 --exp_name  q-10-joint &
python main_bilevel.py --joint --scale 10 --q 50 --exp_name  q-50-joint &
python main_bilevel.py --joint --scale 10 --q 100 --exp_name q-100-joint 
