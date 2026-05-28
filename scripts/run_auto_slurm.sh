#!/bin/bash
# Slurm job: Autonomous/headless simulation using the ttrpg environment.
#
# Usage:
#   sbatch scripts/run_auto_slurm.sh [world] [ticks] [args...]
#   e.g., sbatch scripts/run_auto_slurm.sh tavern 30 --torch
#

#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH -p gpu
#SBATCH -C vram23&sm_70
#SBATCH -t 4:00:00
#SBATCH -o /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/Mark_1/TTRPG/slurm_jobs/run_auto_%j.out

set -euo pipefail

# 1. Load Conda and Activate Environment
module load conda/latest
conda activate /work/pi_andrewlan_umass_edu/nkparikh_umass_edu/environments/ttrpg

# 2. Set directory
CODE_DIR="/work/pi_andrewlan_umass_edu/nkparikh_umass_edu/Mark_1/TTRPG"
cd "${CODE_DIR}"

# 3. Setup caching / directories if needed
export HF_HOME="/scratch4/workspace/nkparikh_umass_edu-compgen/hf_cache"
export HF_DATASETS_CACHE="/scratch4/workspace/nkparikh_umass_edu-compgen/hf_cache"
export PYTHONUNBUFFERED=1

echo "========================================"
echo " Starting TTRPG Autonomous Simulation in Slurm"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Node       : $(hostname)"
echo "  Arguments  : ${@}"
echo "========================================"

# 4. Run the simulation
# Default --debug so Slurm runs always write logs/autonomous_debug_*.log
exec ./scripts/run_auto.sh "${@}" --debug
