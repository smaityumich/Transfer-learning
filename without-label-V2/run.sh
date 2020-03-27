#!/bin/bash

rm slurm-*
python3 json_joiner.py
rm .result/*
rm .data/*
mv result.json result-json/result-np-oracle-bandwidth-coeff-0.4.json
git add --all
git commit -a

