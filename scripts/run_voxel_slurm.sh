#!/bin/bash
# Slurm job: Voxel simulation (Pygame client) using the ttrpg environment.
#
# Usage:
#   sbatch scripts/run_voxel_slurm.sh [world] [args...]
#   e.g., sbatch scripts/run_voxel_slurm.sh voxel_tavern --torch
#

#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH -p gpu
#SBATCH -t 4:00:00
#SBATCH -o /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/Mark_1/TTRPG/slurm_jobs/run_voxel_%j.out

set -euo pipefail

# 1. Load Conda and Activate Environment
module load conda/latest
conda activate /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/environments/ttrpg

# 2. Set directory
CODE_DIR="/work/pi_andrewlan_umass_edu/nkparikh_umass_edu/Mark_1/TTRPG"
cd "${CODE_DIR}"

# 3. Setup headless pygame if running in non-interactive Slurm environment
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

# 4. Setup caching / directories if needed
export HF_HOME="/scratch4/workspace/nkparikh_umass_edu-compgen/hf_cache"
export HF_DATASETS_CACHE="/scratch4/workspace/nkparikh_umass_edu-compgen/hf_cache"
export PYTHONUNBUFFERED=1

echo "========================================"
echo " Starting TTRPG Voxel Simulation in Slurm"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Node       : $(hostname)"
echo "  Arguments  : ${@}"
echo "  Videodriver: ${SDL_VIDEODRIVER}"
echo "========================================"

# 5. Run the voxel simulation
exec ./scripts/run_voxel.sh "${@}"
