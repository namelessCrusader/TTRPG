#!/bin/bash
#SBATCH -c 5  # Number of Cores per Task
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH --mem 30GB
#SBATCH -p gpu-preempt
#SBATCH -t 0-4 # Job time limit
#SBATCH -o /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/Mark_1/TTRPG/slurm_jobs/slurm_%j.out  # %j = job ID

module load conda/latest
conda create --prefix /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/environments/ttrpg -y python=3.13
conda activate /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/environments/ttrpg

cd /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/Mark_1/TTRPG

pip install -r requirements.txt
pip install -r requirements-torch.txt
pip install -r requirements-voxel-gl.txt