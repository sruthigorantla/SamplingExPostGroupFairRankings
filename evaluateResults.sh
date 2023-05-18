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

### for evaluation at top k (--topk) ranks 

python3 main.py --evaluate german_age25 --delta 0.1
python3 main.py --evaluate german_age35 --delta 0.1

python3 main.py --evaluate jee2009_gender --delta 0.1


### for top k evaluation of more than 2 groups 

# python3 main.py --evaluate_multiple_groups german_age --delta 0.05
# python3 main.py --evaluate_multiple_groups jee2009_category --delta 0.1
now=$(date +"%T")
echo "Time after: $now"

cd ../