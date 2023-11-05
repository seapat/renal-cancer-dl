#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/data2/projects/DigiStrudMed_sklein/slurm-test.txt
#SBATCH --nodelist=slurm6,
#SBATCH --mem-per-cpu=1
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=240:00:00



# Activate Anaconda work environment for OpenDrift
source /home/${USER}/.bashrc
source activate sklein 

python ${1} 

# define and create a unique scratch directory
SCRATCH_DIRECTORY=~/scratch/${SLURM_JOBID} 
echo "scratch directory: " ${SCRATCH_DIRECTORY}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

cp ${1} ${SCRATCH_DIRECTORY}

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
python ${1} > output_${SLURM_ARRAY_TASK_ID}.txt

# after the job is done we copy our output back to $SLURM_SUBMIT_DIR
cp output_${SLURM_ARRAY_TASK_ID}.txt ${SLURM_SUBMIT_DIR}

# we step out of the scratch directory and remove it
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}

# happy end
exit 0


#./global/work/${USER}/job-array-example/${SLURM_JOBID}
