#!/bin/sh

echo "Script: train.py"
python train.py | tee train.log

echo "======================="

echo "Script: evaluation.py"
python evaluation.py | tee evaluation.log

echo "======================="

echo "Script: tests/*.py"
pytest tests/ | tee test.log