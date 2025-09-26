#!/bin/bash
#SBATCH --job-name=AP_${1}
#log files:
#SBATCH -e ./logs/run_multimer_jobs_%A_%a_err.txt
#SBATCH -o ./logs/run_multimer_jobs_%A_%a_out.txt
#SBATCH -p low-gn
#SBATCH --gres=gpu:1

#Reserve the entire GPU so no-one else slows you down
#SBATCH --gres=gpu:1

#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH --ntasks=8
#SBATCH --mem=64000
source $HOME/.bashrc
module purge
conda activate ap
cd /home/piccinno/turicci_LMthesis
MAXRAM=$(echo "`ulimit -m` / 1024.0" | bc)
GPUMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tail -1)
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "scale=3; $MAXRAM / $GPUMEM" | bc)
export TF_FORCE_UNIFIED_MEMORY='1'

# CUSTOMIZE THE FOLLOWING SCRIPT PARAMETERS FOR YOUR SPECIFIC TASK:
####
run_multimer_jobs.py \
--mode=pulldown \
--monomer_objects_dir=/home/piccinno/turicci_LMthesis/test_AlphaPulldown/5BV1-2 \
--protein_lists=/home/piccinno/turicci_LMthesis/protein_list_5bv1-1.txt,/home/piccinno/turicci_LMthesis/protein_list_5bv1-2.txt \
--output_path=/home/piccinno/turicci_LMthesis/test_AlphaPulldown/5BV1-2_predictions \
--data_dir=/home/shared/alphafold_db \
--num_cycle=3 \
--job_index=${SLURM_ARRAY_TASK_ID:-0}
####
