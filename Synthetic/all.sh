#!/bin/bash

python main_bilevel.py --flat --change_prob 0.25 --exp_name flat-0.25
python main_bilevel.py --flat --change_prob 0.75 --exp_name flat-0.75
python main_bilevel.py --flat --change_prob 1 --exp_name flat-1
python main_bilevel.py --joint --change_prob 0.25 --exp_name joint-0.25
python main_bilevel.py --joint --change_prob 0.25 --exp_name joint-0.75
python main_bilevel.py --joint --change_prob 1 --exp_name joint-1
