import argparse, os

from evaluation.evaluate import Postprocessing_Evaluator
from evaluation.evaluate_multiple_groups import Postprocessing_Multiple_Evaluator
from processingWithFair.postprocess import Postprocessing


def main():
    # parse command line options
    parser = argparse.ArgumentParser(prog='ALG',
                                     epilog="=== === === end === === ===")

    parser.add_argument("--evaluate",
                        nargs=1,
                        metavar='DATASET',
                        choices=['compas_sex',
                                 'compas_race',
                                 'german_sex',
                                 'german_age25',
                                 'german_age35',
                                 'jee2009_gender',
                                 'biased_normal'],
                        help="evaluates performance and fairness metrics for test DATASET on postprocessing")
    parser.add_argument("--evaluate_multiple_groups",
                        nargs=1,
                        metavar='DATASET',
                        choices=['compas_race_gender',
                                 'german_age',
                                 'german_age_gender',
                                 'jee2009_category',
                                 'jee2009_gender_category'],
                        help="evaluates performance and fairness metrics for test DATASET on postprocessing")
    parser.add_argument("--postprocess",
                        nargs=2,
                        metavar=("DATASET", "method"),
                        choices=['compas',
                                 'german',
                                 'jee2009',
                                 'biased_normal',
                                 'ALG',
                                 'FAIR',
                                 'CELIS',
                                 'EPS_GREEDY',
                                 'LATTICE',
                                 'PREFIX',
                                 'DP',
                                 'RANDOMIZED'],
                        help="reranks all folds for the specified dataset's test fold with FA*IR for pre-processing (alpha = 0.1) or ALG")
    parser.add_argument('--k', type=int,  default=100,
                        help='interval length to audit fairness')
    parser.add_argument('--num_samples', type=int,  default=1000,
                        help='number of rankings to sample from stochastic rankers')
    # parser.add_argument('--topk', type=int,  default=None,
    #                     help='top k ranks to evaluate fairness metrics at')
    parser.add_argument('--consecutive', type=int,  default=None,
                        help='block of \floor{\epsilon k/2} ranks ending at the given value, to evaluate fairness metrics at')
    parser.add_argument('--start', type=int,  default=None,
                        help='starting index of the consecutive ranks')
    parser.add_argument('--rev_flag', type=bool,  default=False,
                        help='set flag to reverse order')
    parser.add_argument('--multi_group', type=bool,  default=False,
                        help='to run ALG on more than 2 groups')  
    parser.add_argument('--eps', type=float,
                        help='epsilon value for fair episolon greedy')     
    parser.add_argument('--delta', type=float,
                        help='delta value for plotting')                
    args = parser.parse_args()

    #################### argparse evaluate ################################

    # args.topk = [100,250,500,750,1000]
    args.topk = [20,40,60,80,100]

    if args.evaluate:
        assert args.delta is not None


    if args.evaluate == ['german_age25']:
        if args.rev_flag:
            resultDir = '../results/GermanCredit/age25_rev/'
        else:
            resultDir = '../results/GermanCredit/age25/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('german_age25',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.consecutive,
                                    args.start,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate()

    elif args.evaluate == ['german_age35']:
        if args.rev_flag:
            resultDir = '../results/GermanCredit/age35_rev/'
        else:
            resultDir = '../results/GermanCredit/age35/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('german_age35',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.consecutive,
                                    args.start,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate()

    elif args.evaluate == ['jee2009_gender']:
        if args.rev_flag:
            resultDir = '../results/JEE2009/gender_rev/'
        else:
            resultDir = '../results/JEE2009/gender/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('jee2009_gender',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.consecutive,
                                    args.start,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate()
    
    elif args.evaluate == ['biased_normal']:
        if args.rev_flag:
            resultDir = '../results/BiasedNormalSynthetic/prot_attr_rev/'
        else:
            resultDir = '../results/BiasedNormalSynthetic/prot_attr/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('biased_normal',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.consecutive,
                                    args.start,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate()
    

    elif args.evaluate == ['german_sex']:
        resultDir = '../results/GermanCredit/gender/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('german_sex',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.consecutive,
                                    args.start,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate()
    
    elif args.evaluate == ['compas_race']:
        if args.rev_flag:
            resultDir = '../results/COMPAS/race_rev/'
        else:
            resultDir = '../results/COMPAS/race/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('compas_race',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.consecutive,
                                    args.start,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate()
    
    elif args.evaluate == ['compas_sex']:
        if args.rev_flag:
            resultDir = '../results/COMPAS/gender_rev/'
        else:
            resultDir = '../results/COMPAS/gender/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('compas_sex',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.consecutive,
                                    args.start,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate()

    #################### argparse evaluate multiple groups ################################
    elif args.evaluate_multiple_groups == ['german_age']:
        resultDir = '../results/GermanCredit/age/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Multiple_Evaluator('german_age',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate_multiple_groups()
    elif args.evaluate_multiple_groups == ['german_age_gender']:
        resultDir = '../results/GermanCredit/age_gender/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Multiple_Evaluator('german_age_gender',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate_multiple_groups()
    elif args.evaluate_multiple_groups == ['jee2009_category']:
        resultDir = '../results/JEE2009/category/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Multiple_Evaluator('jee2009_category',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate_multiple_groups()
    elif args.evaluate_multiple_groups == ['jee2009_gender_category']:
        resultDir = '../results/JEE2009/gender_category/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Multiple_Evaluator('jee2009_gender_category',
                                    resultDir,
                                    binSize,
                                    protAttr,
                                    args.topk,
                                    args.rev_flag,
                                    args.delta)
        evaluator.evaluate_multiple_groups()


    #################### argparse post-process-on-groundtruth ################################
    elif (args.postprocess != None):

        if ('compas' in args.postprocess or \
            'german' in args.postprocess or \
            'jee2009' in args.postprocess or \
            'biased_normal' in args.postprocess) and \
           ('ALG' in args.postprocess \
            or 'FAIR' in args.postprocess \
            or 'CELIS' in args.postprocess \
            or 'EPS_GREEDY' in args.postprocess \
            or 'RANDOMIZED' in args.postprocess \
            or 'LATTICE' in args.postprocess \
            or 'PREFIX' in args.postprocess \
            or 'DP' in args.postprocess):

            postprocessor = Postprocessing(args)
            postprocessor.postprocess()

    else:
        parser.error("choose one command line option")


if __name__ == '__main__':
    main()
