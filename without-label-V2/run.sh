#!/bin/bash

rm slurm-*
python3 json_joiner.py
rm .result/*
rm .data/*
mv result.json result-json/result-nq-unsupervised-bandwidth.json
git add --all
git commit -a

