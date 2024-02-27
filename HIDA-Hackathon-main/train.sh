#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=03:00:00
#SBATCH --reservation=hida

echo AAAA
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76

echo BBB
data_workspace=/hkfs/work/workspace_haic/scratch/qx6387-hida-hackathon-data
group_workspace=/hkfs/work/workspace_haic/scratch/qx6387-WetKoalas

module purge
module load toolkit/nvidia-hpc-sdk/23.9

source ${group_workspace}/HIDA-Hackathon/hida_venv/bin/activate
python ${group_workspace}/HIDA-Hackathon/train.py
