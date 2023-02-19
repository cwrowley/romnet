#!/bin/bash
#SBATCH --job-name=romnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=2:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=gm0796@princeton.edu
module purge
module load anaconda3/2022.5
conda activate /scratch/network/gm0796/nonlinProjEnv
python ./examples/train.py $1 $2
python ./examples/$1.py rom $2
python ./examples/$1.py test $2 savefig
