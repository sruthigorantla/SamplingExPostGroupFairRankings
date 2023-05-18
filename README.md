## Requirements
``python``, ``numpy``, ``matplotlib``




#### To run the experiments
1) ``postprocess.sh``  to perform postprocessing and generate random rankings.

``python3 main.py --postprocess <dataset> <method> --k <top k ranks> --multi_group <True/False>`` 


2) ``evaluateResults.sh``

``python3 main.py --evaluate <dataset_protectedgroup> --topk <top k ranks>`` evaluates the algorithms for the top k ranks, where k is given as input argument ``topk``. 
