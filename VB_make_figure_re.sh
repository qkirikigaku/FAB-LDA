#!/bin/bash
num_doc=$1
num_word=$2
correct_topic=$3
num_K=$4
python VB_matching_improved.py ${num_doc} ${num_word} ${correct_topic} ${num_K}
python VB_data_represent.py ${num_doc} ${num_word} ${correct_topic} ${num_K}