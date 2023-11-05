#!/bin/bash

# Define a list of strings
string_list=("kronecker", "embrace", "cat", "max", "sum", "prod", "attention")

# Loop over the list of strings
for string in "${string_list[@]}"
do
    # Call the Python script and pass the string as an argument
    mamba activate sklein
    nohup accelerate launch --config_file ddp.yaml fusion_train.py --method "$string" --gradient_accumulation_steps 12 --output_dir "/data2/projects/DigiStrudMed_sklein/huggingface/" & 
done
