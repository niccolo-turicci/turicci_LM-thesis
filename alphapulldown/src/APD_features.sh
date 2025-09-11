#!/bin/bash
#SBATCH --job-name=AP_feat
#SBATCH -e logs/create_individual_features_%A_%a_err.txt
#SBATCH -o logs/create_individual_features_%A_%a_out.txt
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --mem=64000

source $HOME/.bashrc
module purge
conda activate alphapulldown
cd /home/piccinno/turicci_LMthesis

####
create_individual_features.py \
--fasta_paths=example_1_sequences.fasta \
--data_dir=/scratch/AlphaFold_DBs/2.3.2 \
--output_dir=/scratch/mydir/test_AlphaPulldown/ \
--max_template_date=2050-01-01 \
--skip_existing=True \
--use_mmseqs2=True \
--seq_index=$SLURM_ARRAY_TASK_ID
#####
