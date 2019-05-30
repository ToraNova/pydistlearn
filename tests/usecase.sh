#!/bin/bash

python3 process_test.py -p sample_jsons/mindem.json -o ta.csv sample_dsets/Demographics10k.csv 
python3 process_test.py -p sample_jsons/minmed.json -o tb.csv sample_dsets/MedicalRecords10k.csv

head -n 5001 ta.csv > ra.csv
head -n 5001 tb.csv > rb.csv
