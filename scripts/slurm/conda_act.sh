#!/bin/bash -l # The -l makes bash act like a login shell, which is needed activate the conda environment 

#SBATCH -J vs_slurm_upload
#SBATCH -o ./out/%j_log.out
#SBATCH --ntasks=1
#SBATCH --array=0-14
FILES=(../workdir/*)

pwd
conda info --envs
conda activate upload