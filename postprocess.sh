#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd src/
now=$(date +"%T")
echo "Time before: $now"


########### reranking
## rerank German Credit dataset folds
# ## for two groups
# python3 main.py --postprocess german ALG --k 100
# python3 main.py --postprocess german EPS_GREEDY --k 100 --eps 0.5 --num_samples 100 
# python3 main.py --postprocess german LATTICE --k 100 --num_samples 100
# python3 main.py --postprocess german DP --k 100 --num_samples 100
# python3 main.py --postprocess german PREFIX --k 100 --num_samples 100

## rerank JEE2009 dataset folds

# ## for two groups
# python3 main.py --postprocess jee2009 ALG --k 100 
# python3 main.py --postprocess jee2009 EPS_GREEDY --k 100 --eps 0.5 --num_samples 100 
# python3 main.py --postprocess jee2009 LATTICE --k 100 --num_samples 100
# python3 main.py --postprocess jee2009 DP --k 100 --num_samples 100 
# python3 main.py --postprocess jee2009 PREFIX --k 100 --num_samples 100

# ## for multiple groups
# python3 main.py --postprocess jee2009 ALG --k 2000 --multi_group=True
# python3 main.py --postprocess jee2009 EPS_GREEDY --k 2000 --eps 0.3 --num_samples 100 --multi_group=True 
# python3 main.py --postprocess jee2009 LATTICE --k 2000 --num_samples 100 --multi_group=True
# python3 main.py --postprocess jee2009 DP --k 2000 --num_samples 100 --multi_group=True
# python3 main.py --postprocess jee2009 PREFIX --k 2000 --num_samples 100 --multi_group=True


now=$(date +"%T")
echo "Time after: $now"
cd ../