    #!/bin/bash
#
#SBATCH --job-name=medical-net
#SBATCH --output=test.out
#SBATCH --nodelist=slurm5,slurm6
##SBATCH --nodes=2
##SBATCH --ntask-per-node=1
##SBATCH --partition=gpu,partition1*
#SBATCH --time=00:24:00
#SBATCH --cpus-per-task=4

source /etc/profile.d/modules.sh
module load anaconda3

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sklein

srun accelerate launch main.py --mixed_precision "bf16" --output_dir "/data2/projects/DigiStrudMed_sklein/huggingface/" --gradient_accumulation_steps 40 --name "no_gather_unfrozen"
