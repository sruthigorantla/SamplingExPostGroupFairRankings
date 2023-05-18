import pandas as pd
import numpy as np
# import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import scipy.stats as stats
import math
import pickle
import copy
import os
from fileinput import filename
from collections import defaultdict, Counter


class Postprocessing_Evaluator():

    def __init__(self, dataset, resultDir, binSize, protAttr, topk, consecutive, start, rev, delta):
        self.__trainingDir = '../data/'
        self.__resultDir = resultDir
        if not os.path.exists(resultDir):
            os.makedirs(resultDir)
        self.__dataset = dataset
        if topk is None:
            self.__k_for_evaluation = start + consecutive - 1
        else:
            self.__k_for_evaluation = topk
        self.rev = rev
        self.__delta = delta

        self.__block = consecutive
        if consecutive is not None:
            self.BLOCK_SIZE = consecutive
        if 'german' in dataset:
            self.__prot_attr_name = self.__dataset.split('_')[1]
            self.__columnNames = ['DurationMonth','CreditAmount','score',self.__prot_attr_name,'query_id','doc_id']

        elif 'biased_normal' in dataset:
            self.__prot_attr_name = 'prot_attr'
            self.__columnNames = ['score',self.__prot_attr_name,'query_id','doc_id']
              
        elif 'compas' in dataset:
            self.__prot_attr_name = self.__dataset.split('_')[1]
            self.__columnNames = ['priors_count','Violence_rawscore','Recidivism_rawscore',self.__prot_attr_name,'query_id','doc_id']

        elif 'jee2009' in dataset:
            self.__prot_attr_name = self.__dataset.split('_')[1]
            self.__columnNames = ['id',self.__prot_attr_name,'mark','query_id','doc_id']

        else:
            self.__prot_attr_name = self.__dataset.split('-')[1]
            if self.__prot_attr_name == 'gender':
                self.__columnNames = ["query_id", "hombre", 'psu_mat', 'psu_len' ,'psu_cie', 'nem' ,'score','doc_id']
            else:
                self.__columnNames = ["query_id", "highschool_type", 'psu_mat', 'psu_len' ,'psu_cie', 'nem' ,'score','doc_id']
        self.__experimentNamesAndFiles = {}
        self.__results = defaultdict(dict)
        self.__result_stds = defaultdict(dict)
        self.axes_dict = {}
        self.figures = {}
        self.y_flag = False

    def collate_results(self, results):

        collated_results = defaultdict(list)
        collated_result_stds = {}

        for result in results:
            for key, val in result.items():
                collated_results[key].append(val)

        for key, val in collated_results.items():
            collated_results[key] = np.mean(val)

            collated_result_stds[key] = np.std(val)

            # print(key, np.mean(val), np.std(val))

        return collated_results, collated_result_stds


    def evaluate(self):
        
        #### choose the directory where reranked results are stored
        if 'german' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'GermanCredit/'
        elif 'biased_normal' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'BiasedNormalSynthetic/'
        elif 'compas' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'COMPAS/'
        elif 'jee2009' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'JEE2009/'
        else:
            raise ValueError("Choose dataset from (enginering/compas/german/jee2009)")
        
        
        #### create empty figures for plots
        mpl.rcParams.update({'font.size': 5, 'lines.linewidth': 5, 'font.family':'CMU Serif'})
        plt.rcParams["axes.grid"] = False
        plt.rcParams['axes.linewidth'] = 1.25
        mpl.rcParams['axes.edgecolor'] = 'k'
        mpl.rcParams["legend.handlelength"] = 6.0
        mpl.rcParams['mathtext.fontset'] = 'stix'

        self.figures['ndcg'] = plt.figure(figsize=(10,10))
        self.axes_dict['ndcg'] = plt.axes()

        self.figures['group_ndcg'] = plt.figure(figsize=(10,10))
        self.axes_dict['group_ndcg'] = plt.axes()

        self.figures['rep'] = plt.figure(figsize=(10,10))
        self.axes_dict['rep'] = plt.axes()

        self.figures['proportion'] = plt.figure(figsize=(20,10))
        self.axes_dict['proportion'] = plt.axes()

        

        #### plot variables
        # EXPERIMENT_NAMES = ['PREFIX','EPS_GREEDY','LATTICE','DP','ALG']#,'EPS_GREEDY'
        EXPERIMENT_NAMES = ['DP','LATTICE','EPS_GREEDY','ALG','PREFIX']#,'EPS_GREEDY'
        metrics = ['ndcg', 'rep', 'pak']

        METRIC_NAMES = ['underranking', 'ndcg', 'group_ndcg', 'mfnr', 'representation', 'rep', 'proportion']
        colormap = {'PREFIX': 'limegreen', 'ALG': 'black', 'FAIR': 'darkorange', 'CELIS': 'deepskyblue', 'RANDOMIZED': 'orange', 'LATTICE': 'orange', 'EPS_GREEDY': 'deepskyblue', 'DP': 'purple'} 
        linemap = {'PREFIX': (0, (3, 1, 1, 1, 1, 1)), 'ALG': (0, (3, 5, 1, 5, 1, 5)), 'FAIR': 'darkorange', 'CELIS': 'deepskyblue', 'RANDOMIZED': 'orange', 'LATTICE': 'solid', 'EPS_GREEDY': 'dashed', 'DP': 'dashdot'} 
        markermap = {'PREFIX': None, 'ALG': None, 'FAIR':None, 'CELIS': None, 'RANDOMIZED': None, 'LATTICE': None, 'EPS_GREEDY': None, 'DP': None}
        markersizemap = {'PREFIX': 25, 'ALG': 25, 'FAIR': 25, 'CELIS': 25, 'RANDOMIZED': 25, 'LATTICE': 25, 'EPS_GREEDY': 25, 'DP': 25}
        markerfillstyle = {'PREFIX':'none', 'ALG': 'full', 'FAIR': 'none', 'CELIS': 'none', 'RANDOMIZED': 'none', 'LATTICE': 'none', 'EPS_GREEDY': 'none', 'DP': 'none'}
        labelmap = {'PREFIX':'Prefix Random walk', 'ALG': 'GDL21', 'FAIR': 'FA*IR', 'CELIS': 'CELIS', 'RANDOMIZED': 'DP', 'LATTICE': 'Random walk', 'EPS_GREEDY': 'Fair $\epsilon$-greedy', 'DP': 'DP'} 
        capsizemap = {'PREFIX':25, 'ALG': 0, 'FAIR': 10, 'CELIS': 10, 'RANDOMIZED': 20, 'LATTICE': 20, 'EPS_GREEDY': 30, 'DP': 15} 

        lns1 = []
        lns2 = []
        lns3 = []
        lns4 = []
        lns5 = []
        lns6 = []
        for experiment in EXPERIMENT_NAMES:
            print(experiment)


            #### get the predictions for the experiment with all the k's

            self.__predictions, self.__groundtruth = self.__prepareData(self.__trainingDir, experiment, self.__prot_attr_name)  
            self.__true_rep_k = []
            
            for top_k in self.__k_for_evaluation:

                if experiment in ["LATTICE", "EPS_GREEDY", "DP", "PREFIX"]:
                    multiple_preds = None
                    multiple_preds = copy.deepcopy(self.__predictions)
                    pred_keys = list(multiple_preds.keys())

                    NUM_SAMPLES = len(multiple_preds[pred_keys[0]])


                #### calculate ndcg only for last item in top-k list
                if 'ndcg' in metrics:
                    print('ndcg')
                    if experiment in ["LATTICE", "EPS_GREEDY", "DP", "PREFIX"]:

                        ndcg = []
                        for ctr in range(NUM_SAMPLES):
                            delta = self.__delta
                            self.__predictions[delta] = multiple_preds[delta][ctr]
                            ndcg.append(self.__ndcg(top_k))
                        self.__results[top_k]['ndcg'], self.__result_stds[top_k]['ndcg'] = self.collate_results(ndcg)

                    else:
                        self.__results[top_k]['ndcg'] = self.__ndcg(top_k)

                        self.__result_stds[top_k]['ndcg'] = {}
                        for key, val in self.__results[top_k]['ndcg'].items():
                            self.__result_stds[top_k]['ndcg'][key]=0.0
                    


                #### calculate representation
                if 'rep' in metrics:
                    print('representation')
                    if experiment in ["DP", "LATTICE", "EPS_GREEDY", "PREFIX"]:

                        representation = []
                        for ctr in range(NUM_SAMPLES):
                            for delta, item in multiple_preds.items():
                                self.__predictions[delta] = multiple_preds[delta][ctr]
                            curr_representation, true_rep_k, self.__true_rep_all = self.__representation(top_k)
                            representation.append(curr_representation)
                        self.__results[top_k]['representation'], self.__result_stds[top_k]['representation'] = self.collate_results(representation)

                    else:
                        self.__results[top_k]['representation'], true_rep_k, self.__true_rep_all = self.__representation(top_k)

                        self.__result_stds[top_k]['representation'] = {}
                        for key, val in self.__results[top_k]['representation'].items():
                            self.__result_stds[top_k]['representation'][key]=0.0
                    

                #### calculate group_ndcg
                if 'group_ndcg' in metrics:
                    if experiment in ["DP", "LATTICE", "EPS_GREEDY", "PREFIX"]:

                        print("group_ndcg")
                        group_ndcg = []
                        for ctr in range(NUM_SAMPLES):
                            for delta, item in multiple_preds.items():
                                self.__predictions[delta] = multiple_preds[delta][ctr]
                            group_ndcg.append(self.__group_ndcg(top_k))
                        self.__results[top_k]['group_ndcg'], self.__result_stds[top_k]['group_ndcg'] = self.collate_results(group_ndcg)

                    else:
                        self.__results[top_k]['group_ndcg']= self.__group_ndcg(top_k)

                        self.__result_stds[top_k]['group_ndcg'] = {}
                        for key, val in self.__results[top_k]['group_ndcg'].items():
                            self.__result_stds[top_k]['group_ndcg'][key]=0.0
                    

                if experiment in ["RANDOMIZED", "LATTICE", "EPS_GREEDY", "DP", "PREFIX"]:
                    self.__predictions = copy.deepcopy(multiple_preds)





                

                #### calculate @k proportions for all ranks for stochastic experiments
                if 'pak' in metrics:
                    if experiment in ["DP", "LATTICE", "EPS_GREEDY", "PREFIX"]:

                        print("@k proportions")
                        self.at_k_proportions = self.__proportions(top_k = self.__k_for_evaluation[-1])
                        
            if 'rep' in metrics:
                lns3 = self.__plot_only_rep('rep', experiment, colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns3)
            if 'ndcg' in metrics:
                lns6 = self.__plot_ndcg('ndcg', experiment, colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns6)
            if 'pak' in metrics and experiment in ["DP", "LATTICE", "EPS_GREEDY", "PREFIX"]:
                lns4 = self.__plot_proportions('proportion', experiment, colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, lns4)
            if 'group_ndcg' in metrics:
                print("here")
                lns5 = self.__plot_group_ndcg('group_ndcg', experiment, colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns5)
            

            

            
            
    def __prepareData(self, pathsToFold, experiment, prot_attr_name):
        '''
        reads training scores and predictions from disc and arranges them NICELY into a dataframe
        '''
        pred_files = list()
        predictedScores = {}
        delta = self.__delta
        for filename in os.listdir(self.__trainingDir):
            if self.rev:
                if ("rev" in filename) and (experiment in filename) and (prot_attr_name in filename):
                    if str(delta) in filename:
                        predictedScores[delta] = pd.read_csv(self.__trainingDir+'/'+filename, sep=",", header=0)
            else:
                if ("rev" not in filename) and (experiment in filename):
                    if 'jee2009' in self.__dataset:
                        if prot_attr_name == "gender" and "category" not in filename:
                            if "RANDOMIZED" in filename or  "PREFIX" in filename or  "LATTICE" in filename or "EPS_GREEDY" in filename or "DP" in filename:
                                if str(delta) in filename:
                                    with open(self.__trainingDir+filename, 'rb') as handle:
                                        predictedScores[delta] = pickle.load(handle)

                            else:
                                delta = float((filename.split('=')[1]).split('.txt')[0])
                                predictedScores[delta] = pd.read_csv(self.__trainingDir+'/'+filename, sep=",", header=0)
                        elif prot_attr_name == 'category_gender' and "category_gender" in filename:
                            if "RANDOMIZED" in filename or  "PREFIX" in filename or  "LATTICE" in filename or "EPS_GREEDY" in filename or "DP" in filename:
                                if str(delta) in filename:
                                    with open(self.__trainingDir+filename, 'rb') as handle:
                                        predictedScores[delta] = pickle.load(handle)

                            else:
                                if str(delta) in filename:
                                    predictedScores[delta] = pd.read_csv(self.__trainingDir+'/'+filename, sep=",", header=0)
                    elif prot_attr_name in filename:
                        if "RANDOMIZED" in filename or  "LATTICE" in filename or "PREFIX" in filename or "EPS_GREEDY" in filename or "DP" in filename:
                            if str(delta) in filename:
                                with open(self.__trainingDir+filename, 'rb') as handle:
                                    predictedScores[delta] = pickle.load(handle)
                        else:
                            if str(delta) in filename:
                                predictedScores[delta] = pd.read_csv(self.__trainingDir+'/'+filename, sep=",", header=0)

        if 'german' in self.__dataset:
            groundtruth = pd.read_csv(self.__trainingDir+'/'+'GermanCredit_'+prot_attr_name+'.csv', sep=",", header=0)
            if self.rev:
                groundtruth['score'] = groundtruth['score'].apply(lambda val: 1-val)
            groundtruth = (groundtruth.sort_values(by=['score'], ascending=False)).reset_index(drop=True)
            
        elif 'biased_normal' in self.__dataset:
            groundtruth = pd.read_csv(self.__trainingDir+'/'+'BiasedNormalSynthetic_'+prot_attr_name+'.csv', sep=",", header=0)
            if self.rev:
                groundtruth['score'] = groundtruth['score'].apply(lambda val: 1-val)
            groundtruth = (groundtruth.sort_values(by=['score'], ascending=False)).reset_index(drop=True)
        elif 'compas' in self.__dataset:
            groundtruth = pd.read_csv(self.__trainingDir+'/'+'ProPublica_'+prot_attr_name+'.csv', sep=",", header=0)
            if not self.rev:
                groundtruth['Recidivism_rawscore'] = groundtruth['Recidivism_rawscore'].apply(lambda val: 1-val)
            groundtruth = (groundtruth.sort_values(by=['Recidivism_rawscore'], ascending=False)).reset_index(drop=True)
        elif 'jee2009' in self.__dataset:
            groundtruth = pd.read_csv(self.__trainingDir+'/'+'jee2009_'+prot_attr_name+'.csv', sep=",", header=0)
            if self.rev:
                groundtruth['mark'] = groundtruth['mark'].apply(lambda val: 1-val)
            groundtruth = (groundtruth.sort_values(by=['mark'], ascending=False)).reset_index(drop=True)
        
        groundtruth['doc_id'] = np.arange(len(groundtruth))+1
        return predictedScores, groundtruth

    def __ndcg(self, top_k):
        '''
        calculate ndcg in top-k for all the deltas for the given experiment
        
        ndcg@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}
        
        '''
        
        ### calculate maximum ndcg possible
        
        idcg = 0
        data = self.__groundtruth
        for i in range(top_k):
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                try:
                    score = data.loc[data['doc_id'] == i+1, 'score'].iloc[0]
                except IndexError:
                    score = 0
            elif 'compas' in self.__dataset:
                score = data.loc[data['doc_id'] == i+1, 'Recidivism_rawscore'].iloc[0]
            elif 'jee2009' in self.__dataset:
                score = data.loc[data['doc_id'] == i+1, 'mark'].iloc[0]
                score += 86
                score /= 510.0
            idcg += (2**score-1)/(np.log(i+2))
        ndcg_results = {}
        ### calculate dcg in the top k predicted ranks
        for delta, preds in self.__predictions.items():
            preds_k = preds.head(top_k)
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                scores = preds_k['score'].reset_index(drop=True).to_numpy()
            elif 'compas' in self.__dataset:
                scores = preds_k['Recidivism_rawscore'].reset_index(drop=True).to_numpy()
            elif 'jee2009' in self.__dataset:
                scores = preds_k['mark'].reset_index(drop=True).to_numpy()
                scores += 86
                scores /= 510.0
            dcg = 0
            for i in range(top_k):
                dcg += (2**scores[i]-1)/(np.log(i+2))
            ndcg_results[delta] = dcg/idcg
        return ndcg_results  

    def __ndcg2(self,top_k):
        true_dcg = 0
        data = self.__groundtruth
        for i in range(top_k):
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                scorer = 'score'
                score = data.loc[i][scorer]
            elif 'jee2009' in self.__dataset:  
                scorer = 'mark'
                score = data.loc[i][scorer]
                score += 86
                score /= 510.0
            
            true_dcg += (2**score-1)/(np.log(i+2))
            

        ### calculate dcg in the top k predicted ranks
        ndcg_results = {}
        delta = self.__delta
        preds_k = self.__predictions[delta].head(top_k)
        pred_dcg = 0
        for i in range(top_k):
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                scorer = 'score'
                score = preds_k.loc[i][scorer]
            elif 'jee2009' in self.__dataset:  
                scorer = 'mark'
                score = preds_k.loc[i][scorer]
                score += 86
                score /= 510.0

            
            pred_dcg += (2**score-1)/(np.log(i+2))
        ndcg_results[delta] = pred_dcg/true_dcg
        return ndcg_results
    
    def __ndcg_old(self,top_k):
        true_dcg = 0
        true_count = 0
        data = self.__groundtruth
        for i in range(top_k):
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                scorer = 'score'
                score = data.loc[i][scorer]
            elif 'jee2009' in self.__dataset:  
                scorer = 'mark'
                score = data.loc[i][scorer]
                score += 86
                score /= 510.0
            true_count += 1

            true_dcg += (2**score-1)/(np.log(i+2))
            
        if true_count > 0:
            true_dcg /= true_count
        
        ### calculate dcg in the top k predicted ranks
        ndcg_results = {}
        delta = self.__delta
        preds_k = self.__predictions[delta].head(top_k)
        pred_dcg = 0
        pred_count = 0
        for i in range(top_k):
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                scorer = 'score'
                score = preds_k.loc[i][scorer]
            elif 'jee2009' in self.__dataset:  
                scorer = 'mark'
                score = preds_k.loc[i][scorer]
                score += 86
                score /= 510.0
            pred_count += 1

            
            pred_dcg += (2**score-1)/(np.log(i+2))
        if pred_count > 0:
            pred_dcg /= pred_count
        ndcg_results[delta] = pred_dcg/true_dcg
        return ndcg_results
    
    def __representation(self, top_k):
        '''
        calculate representation of the protected group in top-k for all the deltas for the given experiment
        
        representationProt@k = #protected@k/ k

        '''

        #### calculate groundtruth representation
        if self.__block is not None:
            true_data = self.__groundtruth
            true_data_k = self.__groundtruth.iloc[self.__block - self.BLOCK_SIZE : self.__block]
            true_rep_k = float(len(true_data_k.loc[true_data_k[self.__prot_attr_name] == 1.0]))/self.BLOCK_SIZE
            true_rep_all = float(len(true_data.loc[true_data[self.__prot_attr_name] == 1.0]))/len(true_data)

            representation_results = {}
            for delta, preds in self.__predictions.items():
                preds_k = preds.iloc[self.__block - self.BLOCK_SIZE : self.__block]
                s = len(preds_k.loc[preds_k[self.__prot_attr_name] == 1.0])
                representation_results[delta] = float(s)/self.BLOCK_SIZE
                
            return representation_results, true_rep_k, true_rep_all
        elif top_k is not None:
            true_data = self.__groundtruth
            true_data_k = self.__groundtruth.head(top_k)
            true_rep_k = float(len(true_data_k.loc[true_data_k[self.__prot_attr_name] == 1.0]))/top_k
            true_rep_all = float(len(true_data.loc[true_data[self.__prot_attr_name] == 1.0]))/len(true_data)

            representation_results = {}
            for delta, preds in self.__predictions.items():
                preds_k = preds.head(top_k)
                # print(preds_k.keys())
                s = len(preds_k.loc[preds_k[self.__prot_attr_name] == 1.0])
                representation_results[delta] = float(s)/top_k
                
            return representation_results, true_rep_k, true_rep_all


    def __group_ndcg(self, top_k):
        '''
        calculate representation of the protected group in top-k for all the deltas for the given experiment
        
        representationProt@k = #protected@k/ k

        '''

                
        true_dcg0 = 0
        true_dcg1 = 0
        true_count0 = 0
        true_count1 = 0
        data = self.__groundtruth
        for i in range(top_k):
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                scorer = 'score'
            elif 'jee2009' in self.__dataset:  
                scorer = 'mark'
            
            score0 = 0
            score1 = 0
            if data.loc[i][self.__prot_attr_name] == 0:
                score0 = data.loc[i][scorer]
                if 'jee2009' in self.__dataset:  
                    score0 += 86
                    score0 /= 510.0
                true_count0 += 1

            elif data.loc[i][self.__prot_attr_name] == 1:
                score1 = data.loc[i][scorer]
                if 'jee2009' in self.__dataset:  
                    score1 += 86
                    score1 /= 510.0
                true_count1 += 1
            else:
                print('fail', i)
                exit()
            true_dcg0 += (2**score0-1)/(np.log(i+2))
            true_dcg1 += (2**score1-1)/(np.log(i+2))
        
        # if true_count0 > 0:
        #     true_dcg0 /= true_count0
        # if true_count1 > 0:
        #     true_dcg1 /= true_count1
        group_ndcg_results = {}
        
        ### calculate dcg in the top k predicted ranks
        for delta, preds in self.__predictions.items():
            preds_k = preds.head(top_k)
            pred_dcg0 = 0
            pred_dcg1 = 0
            # pred_count0 = 0
            # pred_count1 = 0
            for i in range(top_k):
                if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                    scorer = 'score'
                elif 'jee2009' in self.__dataset:  
                    scorer = 'mark'
                
                score0 = 0
                score1 = 0
                if preds_k.loc[i][self.__prot_attr_name] == 0:
                    score0 = preds_k.loc[i][scorer]
                    if 'jee2009' in self.__dataset:  
                        score0 += 86
                        score0 /= 510.0
                    # pred_count0 += 1

                elif preds_k.loc[i][self.__prot_attr_name] == 1:
                    score1 = preds_k.loc[i][scorer]
                    if 'jee2009' in self.__dataset:  
                        score1 += 86
                        score1 /= 510.0
                    # pred_count1 += 1
                else:
                    print('fail', i)
                    exit()
                pred_dcg0 += (2**score0-1)/(np.log(i+2))
                pred_dcg1 += (2**score1-1)/(np.log(i+2))
                # print(pred_dcg0,pred_dcg1)
            # if pred_count0 > 0:
            #     pred_dcg0 /= pred_count0
            # if pred_count1 > 0:
            #     pred_dcg1 /= pred_count1
            
            group_ndcg_results[delta] = min(pred_dcg0, pred_dcg1)/max(true_dcg0, true_dcg1)
            # print(pred_dcg0, pred_dcg1, true_dcg0, true_dcg1)
            # exit()
            
        return group_ndcg_results


    def __proportions(self, top_k):

        proportion_results = {}
        for delta, preds in self.__predictions.items():
            ans = []
            for rank in range(top_k):

                # get the prot attributes across all samples in this rank
                prot_attrs = []
                for pred in preds:
                    prot_attrs.append(pred.iloc[rank][self.__prot_attr_name])
                    assert pred.iloc[rank][self.__prot_attr_name] == list(pred[self.__prot_attr_name])[rank]

                ans.append(Counter(prot_attrs)[1.0] / sum([val for key, val in Counter(prot_attrs).items()]))

            proportion_results[delta] = ans

        return proportion_results





    
    
    def __plot_only_rep(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns=None):
        
        # Print the values from delta = self.__delta

        p = self.__true_rep_all
        p_k = self.__true_rep_k        
                
        all_ks = list(self.__results.keys())


        result1 = np.asarray([self.__results[k]['representation'][self.__delta] for k in all_ks])
        result_std1 = np.asarray([self.__result_stds[k]['representation'][self.__delta] for k in all_ks])


        
        if self.__block is not None:
            k_str = '$k\' = $'+str(self.__block - self.BLOCK_SIZE+1)+' to '+str(self.__block)
        else:
            k_str = '$k = $'+str(self.__k_for_evaluation[-1])
        if self.__prot_attr_name == 'sex' or self.__prot_attr_name == 'gender':
            self.axes_dict[metric1].set_title('Protected group = female, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age35':
            self.axes_dict[metric1].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age25':
            self.axes_dict[metric1].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'race':
            self.axes_dict[metric1].set_title('Protected group = African American, '+k_str, fontsize=40)
        else:
            self.axes_dict[metric1].set_title('Protected group = '+self.__prot_attr_name+', '+k_str, fontsize=40)
        self.axes_dict[metric1].set_xlabel('top $i$ ranks', fontsize=40)
        self.axes_dict[metric1].set_ylabel('representation', fontsize=40)
        self.axes_dict[metric1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=40)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=40)
        self.axes_dict[metric1].set_facecolor("white")
        # self.axes_dict[metric1].set_ylim(-0.05,0.45)
        self.axes_dict[metric1].set_xlim(20,100)
        
        
        

        ##### , fillstyle='none',markersize=3
        markerwidthval = 2
        if "FAIR" in experiment:
            name = 'FA*IR'
            lns1 = self.axes_dict[metric1].errorbar((all_ks), result1, result_std1, capsize=capsizemap[experiment], color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
        else:
            name = experiment
            if "ALG" in experiment or 'CELIS' in experiment:
                lns1 = self.axes_dict[metric1].plot(all_ks, result1, color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            else:
                lns1 = self.axes_dict[metric1].plot(np.array(all_ks), result1, color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
                self.axes_dict[metric1].fill_between(x=np.array(all_ks), y1=result1-result_std1, y2=result1+result_std1, color=colormap[experiment],alpha=0.35)
            if metric1 == 'rep' and not self.y_flag:
                self.axes_dict[metric1].errorbar(all_ks, [p]*len(all_ks),  label='y = ${p^*}$', color='red', linestyle='dotted', linewidth=10.0)
                self.y_flag = True
                
        #----- uncomment this if you want legend
        # self.axes_dict[metric1].legend(prop={'size': 50}, facecolor='white', loc='upper center',ncol=6, framealpha=1)#, bbox_to_anchor=(0.49, 1.1))

        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.pdf',bbox_inches='tight')

        return None

    def __plot_group_ndcg(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns=None):
        
        # Print the values from delta = self.__delta

        all_ks = list(self.__results.keys())


        result1 = np.asarray([self.__results[k]['group_ndcg'][self.__delta] for k in all_ks])
        result_std1 = np.asarray([self.__result_stds[k]['group_ndcg'][self.__delta] for k in all_ks])


        
        if self.__block is not None:
            k_str = '$k\' = $'+str(self.__block - self.BLOCK_SIZE+1)+' to '+str(self.__block)
        else:
            k_str = '$k = $'+str(self.__k_for_evaluation[-1])
        if self.__prot_attr_name == 'sex' or self.__prot_attr_name == 'gender':
            self.axes_dict[metric1].set_title('Protected group = female, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age35':
            self.axes_dict[metric1].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age25':
            self.axes_dict[metric1].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'race':
            self.axes_dict[metric1].set_title('Protected group = African American, '+k_str, fontsize=40)
        else:
            self.axes_dict[metric1].set_title('Protected group = '+self.__prot_attr_name+', '+k_str, fontsize=40)
        self.axes_dict[metric1].set_xlabel('$k\'$', fontsize=40)
        self.axes_dict[metric1].set_ylabel('group_ndcg', fontsize=40)
        self.axes_dict[metric1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=40)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=40)
        self.axes_dict[metric1].set_facecolor("white")
        self.axes_dict[metric1].set_ylim(0.5,1.05)
        self.axes_dict[metric1].set_xlim(20,100)


        ##### , fillstyle='none',markersize=3
        markerwidthval = 4
        if "FAIR" in experiment:
            name = 'FA*IR'
            lns1 = self.axes_dict[metric1].errorbar((all_ks), result1, result_std1, capsize=capsizemap[experiment], color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
        else:
            name = experiment
            if "ALG" in experiment or 'CELIS' in experiment:
                lns1 = self.axes_dict[metric1].plot(all_ks, result1, color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            else:
                lns1 = self.axes_dict[metric1].plot(all_ks, result1, color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
                self.axes_dict[metric1].fill_between(x=np.array(all_ks), y1=result1-result_std1, y2=result1+result_std1, color=colormap[experiment],alpha=0.35)

        #----- uncomment this if you want legend
        # self.axes_dict[metric1].legend(prop={'size': 50}, facecolor='white', loc='upper center',ncol=4, framealpha=1)#, bbox_to_anchor=(0.49, 1.1))

        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.pdf',bbox_inches='tight')

        return None

    def __plot_ndcg(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns=None):
        
        # Print the values from delta = self.__delta

        all_ks = list(self.__results.keys())


        result1 = np.asarray([self.__results[k]['ndcg'][self.__delta] for k in all_ks])
        result_std1 = np.asarray([self.__result_stds[k]['ndcg'][self.__delta] for k in all_ks])


        
        if self.__block is not None:
            k_str = '$k\' = $'+str(self.__block - self.BLOCK_SIZE+1)+' to '+str(self.__block)
        else:
            k_str = '$k = $'+str(self.__k_for_evaluation[-1])
        if self.__prot_attr_name == 'sex' or self.__prot_attr_name == 'gender':
            self.axes_dict[metric1].set_title('Protected group = female, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age35':
            self.axes_dict[metric1].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age25':
            self.axes_dict[metric1].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'race':
            self.axes_dict[metric1].set_title('Protected group = African American, '+k_str, fontsize=40)
        else:
            self.axes_dict[metric1].set_title('Protected group = '+self.__prot_attr_name+', '+k_str, fontsize=40)
        self.axes_dict[metric1].set_xlabel('top $i$ ranks', fontsize=40)
        self.axes_dict[metric1].set_ylabel('ndcg', fontsize=40)
        self.axes_dict[metric1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=40)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=40)
        self.axes_dict[metric1].set_facecolor("white")
        self.axes_dict[metric1].set_ylim(0.95,1.01)
        self.axes_dict[metric1].set_xlim(20,100)

        ##### , fillstyle='none',markersize=3
        markerwidthval = 4
        name = experiment
        lns1 = self.axes_dict[metric1].plot(all_ks, result1, color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
        if experiment in ["LATTICE", "EPS_GREEDY", "DP", "PREFIX"]:
            self.axes_dict[metric1].fill_between(x=np.array(all_ks), y1=result1-result_std1, y2=result1+result_std1, color=colormap[experiment],alpha=0.35)
    
        #----- uncomment this if you want legend
        # self.axes_dict[metric1].legend(prop={'size': 50}, facecolor='white', loc='upper center',ncol=4, framealpha=1)#, bbox_to_anchor=(0.49, 1.1))

        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.pdf',bbox_inches='tight')

        return lns1

    def __plot_proportions(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, lns=None):
        
        # Plots the proportion of groups for stochastic ranks in each rank
        # Values are taken from the delta specified in self.__delta

        result1 = self.at_k_proportions[self.__delta]
        all_ks = [val for val in range(1,self.__k_for_evaluation[-1]+1)]

        assert len(result1) == len(all_ks)
        
        if self.__block is not None:
            k_str = '$k\' = $'+str(self.__block - self.BLOCK_SIZE+1)+' to '+str(self.__block)
        else:
            k_str = '$k = $'+str(self.__k_for_evaluation[-1])

        if self.__prot_attr_name == 'sex' or self.__prot_attr_name == 'gender':
            self.axes_dict[metric1].set_title('Protected group = female, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age35':
            self.axes_dict[metric1].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age25':
            self.axes_dict[metric1].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'race':
            self.axes_dict[metric1].set_title('Protected group = African American, '+k_str, fontsize=40)
        else:
            self.axes_dict[metric1].set_title('Protected group = '+self.__prot_attr_name+', '+k_str, fontsize=40)
        
        self.axes_dict[metric1].set_xlabel('at rank $i$', fontsize=40)
        self.axes_dict[metric1].set_ylabel('fraction of rankings', fontsize=40)
        self.axes_dict[metric1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=40)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=40)
        self.axes_dict[metric1].set_facecolor("white")
        # self.axes_dict[metric1].set_ylim(-0.05,0.65)
        
        
        markerwidthval = 4

        name = experiment
        lns1 = self.axes_dict[metric1].plot(all_ks, result1, color=colormap[experiment], label=labelmap[experiment], linestyle=linemap[experiment])

        self.axes_dict[metric1].plot(all_ks, [self.__true_rep_all]*len(all_ks),  label='y = ${p^*}$', color='red', linestyle='dotted', linewidth=10.0)
        #---- uncomment this if you want legend
        # self.axes_dict[metric1].legend( prop={'size': 20}, facecolor='white', loc='upper center',ncol=3, framealpha=1)#, bbox_to_anchor=(0.49, 1.1))

        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.pdf',bbox_inches='tight')
