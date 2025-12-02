#!/bin/bash
#SBATCH --job-name=match_filtering_run
#SBATCH --output=m.out
#SBATCH --licenses=sps
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6GB
#SBATCH --time=4-00:00

echo "-------------- Running my job ----------------"

echo " Working dir: $PWD"
python3 match_filtering.py --b bbh_gwfs --j bbh_config.json
echo "-------------- End of  my job ----------------"

exit 0
