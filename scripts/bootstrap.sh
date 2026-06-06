#!/bin/bash
set -e
.build/release/orion generate -o trainingdata/bootstrap -n 5000 -a heuristic --monte-carlo-samples 500 -t 1.50 -b 128
.build/release/orion train -i trainingdata/bootstrap.bin.lz4 -e 100 -b 128 -o models/model_c0_e100_b128 --learning-rate 0.0003 --weight-decay 0.0 --early-stopping 10 --dropout 0.1
.build/release/orion play -n 500 -a models/model_c0_e100_b128/ random -t 0.00
