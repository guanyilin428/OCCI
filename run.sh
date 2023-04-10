#!/bin/bash

#SBATCH -J occi-inst              # The job name
#SBATCH -o ret_inst.out           # Write the standard output to file named 'ret.out'
#SBATCH -e ret_inst.err           # Write the standard error to file named 'ret.err'

#- Resources

#SBATCH -t 1-23:59:00                # Run for a maximum time of 1 days, 20 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:4                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-long               # Request QOS Type
#SBATCH --exclude=gpu-v02,gpu-v07,gpu-t12,gpu-t13


#- Log information

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#-Load environments
source /tools/module_env.sh
module list             # list modules loaded

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load vim/8.1.2424

##- language
module load python3/3.6.8

##- virtualenv
source /home/S/guanyilin/anaconda3/bin/activate occi

echo $(module list)              # list modules loaded
echo $(which python3)

#- Job step
python test.py

#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"