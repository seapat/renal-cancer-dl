#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/data2/projects/DigiStrudMed_sklein/slurm-test.txt
#SBATCH --nodelist=slurm6,
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=300000:00:00

# define and create a unique scratch directory
SCRATCH_DIRECTORY=/global/work/${USER}/kelp/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

module

# Activate Anaconda environment
source /home/${USER}/.bashrc
source activate sklein 

# we execute the job and time it
echo success
