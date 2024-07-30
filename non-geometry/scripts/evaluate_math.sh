#!/bin/bash
set -x

# Specify your EXEHOME first.
cd ${EXEHOME}

# Specify the input path
INPUTPATH=''

python evaluate.py --input_path ${INPUTPATH}\
                --dataset_type 'math'

