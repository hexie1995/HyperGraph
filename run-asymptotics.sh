#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=serial_test # Job name
#SBATCH -o logs/asymptotics-log.out
# Standard output and error log
#SBATCH --mail-user=pchodrow@middlebury.edu
# Where to send mail
#SBATCH --mail-type=ALL
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=50gb
# Job memory request 
#SBATCH --partition=standard
# Partition (queue)
#SBATCH --time=144:00:00
# Time limit hrs:min:sec 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 
echo "Starting: "`date +"%D %T"` 
# perform compute

export PYTHONUNBUFFERED=TRUE
~/.conda/envs/hypermech/bin/python -u code/asymptotics.py

# make all
# End of job info 
echo "Ending: "`date +"%D %T"`