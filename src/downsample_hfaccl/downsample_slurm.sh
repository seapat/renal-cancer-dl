#!/bin/bash
#
#SBATCH --job-name=medical-net
#SBATCH --output=test.out
#SBATCH --nodelist=slrum1,slurm2,slurm3,slurm4,slurm5,slurm6
##SBATCH --nodes=6
##SBATCH --ntask-per-node=1
##SBATCH --partition=gpu,partition1*
#SBATCH --time=00:24:00
#SBATCH --cpus-per-task=4

source /etc/profile.d/modules.sh
module load anaconda3

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sklein

srun accelerate launch --config_file ./default_config.yaml --multi_gpu --gpu_ids 0,1,2,3 --num_processes 4 hugg_downsample.py --mixed_precision "bf16" --output_dir "/data2/projects/DigiStrudMed_sklein/huggingface/" --gradient_accumulation_steps 40 --job_type "onecycle" --run_name "ddp"
