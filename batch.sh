#!/bin/bash -l
#SBATCH --verbose
#SBATCH --partition=gpuq
#SBATCH --nodes 1
#SBATCH --job-name=fuzzy5
#SBATCH --gpus=1
#SBATCH --output=/mnt/beegfs/hmurali/ML/data/fuzzy5.out
#SBATCH --error=/mnt/beegfs/hmurali/ML/data/fuzzy5.err
#SBATCH --time 24:00:00

#python IKKTtorch.py READIN SAVE NCOL Niters g NMAT namePrefix [ "(deltaTStepLeapFrog, nsteps)" ]

# python -u main.py --fresh --name spin0Init --niters 400 --ncol 45 --coupling 100 --omega 1 --pIKKT-type 2 --force --step-size 1 --save --nsteps 100 --spin 0
python -u main.py --fresh --name spin0Init --niters 400 --ncol 40 --coupling 150 --omega 1 --pIKKT-type 2 --force --step-size 1 --save --spin 0
# python -u main.py --fresh --name spin0Init --niters 400 --ncol 45 --coupling 10 --omega 1 --pIKKT-type 2 --force --step-size 1 --save --spin 0

# python -u main.py --fresh --name pIkkt --niters 20 --ncol 40 --coupling 20 --omega 1 --pIKKT-type 2 --force --save
# python -u main.py --fresh --name check --niters 300 --ncol 40 --coupling 20 --omega 1 --pIKKT-type 2 --force --save --dry-run

# for g in {20..220..20}
# do
#     python -u main.py --fresh --name pIkkt --niters 300 --ncol 50 --coupling $g --omega 1 --pIKKT-type 2 --force --save --step-size 3 --nsteps 300
# done
