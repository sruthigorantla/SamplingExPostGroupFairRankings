import pandas as pd
import numpy as np
# import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import math
import os
import copy
import pickle
from fileinput import filename
from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict, Counter


class Postprocessing_Multiple_Evaluator():
    
    def __init__(self, dataset, resultDir, binSize, protAttr, topk, rev, delta):
        self.__trainingDir = '../data/'
        self.__resultDir = resultDir
        if not os.path.exists(resultDir):
            os.makedirs(resultDir)
        self.__dataset = dataset
        self.__k_for_evaluation = topk
        self.rev = rev
        self.__delta = delta
        if 'german' in dataset:
            name = self.__dataset.split('_')
            if len(name) > 2:
                self.__prot_attr_name = name[1]+'_'+name[2]
            elif len(name) == 2:
                self.__prot_attr_name = name[1]
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
            self.__groupname = {0:'GE',1:'SC',2:'ST',3:'OC',4:'ON'}
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



    def collate_results(self, results):

        collated_results = defaultdict(list)
        collated_result_stds = {}


        for result in results:
            for key, val in result.items():
                collated_results[key].append(val)

        for key, val in collated_results.items():

            collated_results[key] = np.mean(np.asarray(val), axis=0)
            collated_result_stds[key] = np.std(np.asarray(val), axis=0)

        return collated_results, collated_result_stds




    def evaluate_multiple_groups(self):
        
        #### choose the directory where reranked results are stored
        if 'german' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'GermanCredit/'
            self.__num_groups = 3
        elif 'jee2009' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'JEE2009/'
            self.__num_groups = 5
        else:
            raise ValueError("Choose dataset from (enginering/compas/german)")
        
        
        #### create empty figures for plots
        # mpl.rcParams.update({'font.size': 5, 'lines.linewidth': 4, 'lines.markersize': 40, 'font.family':'CMU Serif'})
        mpl.rcParams.update({'font.size': 5, 'lines.linewidth': 7.5, 'font.family':'CMU Serif'})
        plt.rcParams["axes.grid"] = False
        plt.rcParams['axes.linewidth'] = 1.25
        mpl.rcParams['axes.edgecolor'] = 'k'
        mpl.rcParams["legend.handlelength"] = 6.0
        mpl.rcParams['mathtext.fontset'] = 'stix'


        self.figures['ndcg'] = plt.figure(figsize=(10,10))
        self.axes_dict['ndcg'] = plt.axes()

        self.figures['group_ndcg'], self.axes_dict['group_ndcg'] = plt.subplots(self.__num_groups, figsize=(10,10))

        self.figures['rep'] = {}
        self.figures['proportion'] = {}
        self.axes_dict['rep'] = {}
        self.axes_dict['proportion'] = {}
        for j in range(self.__num_groups):
            self.figures['rep'][j] = plt.figure(figsize=(10,10))
            self.axes_dict['rep'][j] = plt.axes()
            self.figures['proportion'][j] = plt.figure(figsize=(20,10))
            self.axes_dict['proportion'][j] = plt.axes()

        #### plot variables
        EXPERIMENT_NAMES = ['DP','LATTICE','EPS_GREEDY','ALG','PREFIX']#,'EPS_GREEDY'
        metrics = ['rep','pak']#['ndcg', 'rep', 'pak']

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
                        if prot_attr_name == "gender" and "category_gender" not in filename:
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
                        elif prot_attr_name == 'category' and "gender" not in filename:
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

    def __ndcgold(self, top_k):
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
                score /= 424.0
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
                scores /= 424.0
            dcg = 0
            for i in range(top_k):
                dcg += (2**scores[i]-1)/(np.log(i+2))
            ndcg_results[delta] = dcg/idcg

        return ndcg_results   
    

    def __representation(self, top_k):
        '''
        calculate representation of the protected group in top-k for all the deltas for the given experiment
        
        representationProt@k = #protected@k/ k
        '''

        #### calculate groundtruth representation
        true_data = self.__groundtruth
        NUM_GROUPS = len(pd.unique(true_data[self.__prot_attr_name]))
        assert NUM_GROUPS == self.__num_groups

        true_data_k = self.__groundtruth.head(top_k)
        true_rep_k = np.zeros(NUM_GROUPS)
        true_rep_all = np.zeros(NUM_GROUPS)
        
        for i in range(NUM_GROUPS): 
            true_rep_k[i] = float(len(true_data_k.loc[true_data_k[self.__prot_attr_name] == i]))/top_k
            true_rep_all[i] = float(len(true_data.loc[true_data[self.__prot_attr_name] == i]))/len(true_data)

        representation_results = {}
        for delta, preds in self.__predictions.items():

            preds_k = preds.head(top_k)
            
            ratios = np.zeros(NUM_GROUPS)
            for i in range(NUM_GROUPS):
                ratios[i] = len(preds_k.loc[preds_k[self.__prot_attr_name] == i])/(top_k)

            representation_results[delta] = ratios
        
        return representation_results, true_rep_k, true_rep_all


    def __proportions(self, top_k):

        # Calculates the proportions for each group

        proportion_results = {}
        for delta, preds in self.__predictions.items():
            ans = []
            for rank in range(top_k):

                # get the prot attributes across all samples in this rank
                prot_attrs = []
                for pred in preds:
                    prot_attrs.append(pred.iloc[rank][self.__prot_attr_name])
                    assert pred.iloc[rank][self.__prot_attr_name] == list(pred[self.__prot_attr_name])[rank]

                denominator = sum([val for key, val in Counter(prot_attrs).items()])
                ans.append([Counter(prot_attrs)[float(group)]/denominator for group in range(self.__num_groups)])

            proportion_results[delta] = ans

        return proportion_results



    def __plot_with_delta_for_x(self, metric1, metric2, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns=None):
        
        p = self.__true_rep_all      
                
        
        all_ks = list(self.__results.keys())

        all_deltas = list(self.__results[all_ks[-1]][metric1].keys())
        deltas = np.sort(all_deltas)



        result1 = [self.__results[all_ks[-1]][metric1][delta] for delta in all_deltas]
        result_std1 = [self.__result_stds[all_ks[-1]][metric1][delta] for delta in all_deltas]



        k_str = '$k = $'+str(self.__k_for_evaluation[-1])
        if self.__prot_attr_name == 'sex':
            self.axes_dict[metric1].set_title('Protected group = female, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age35':
            self.axes_dict[metric1].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'age25':
            self.axes_dict[metric1].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=40)
        elif self.__prot_attr_name == 'race':
            self.axes_dict[metric1].set_title('Protected group = African American, '+k_str, fontsize=40)
        else:
            self.axes_dict[metric1].set_title('group based on '+self.__prot_attr_name+', '+k_str, fontsize=40)
        self.axes_dict[metric1].set_xlabel('$\delta$', fontsize=40)
        self.axes_dict[metric1].set_ylabel(metric1, fontsize=40)
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=40)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=40)
        self.axes_dict[metric1].set_facecolor("white")
        

        # self.axes_dict[metric1].legend(loc="lower right", prop={'size': 20}, facecolor='white', framealpha=1)


        markerwidthval = 4



        name = experiment
        if "ALG" in experiment or 'CELIS' in experiment:
            lns1 = self.axes_dict[metric1].errorbar(deltas, result1, result_std1, capsize=capsizemap[experiment], color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
        else:
            lns1 = self.axes_dict[metric1].errorbar(deltas, result1, result_std1, capsize=capsizemap[experiment], color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)

        
        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.pdf',bbox_inches='tight')

        return None


    def __plot_only_rep(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns=None):
        

        p = self.__true_rep_all 
        NUM_GROUPS = len(p)
                
        all_ks = list(self.__results.keys())

        groupwise_results = {}
        groupwise_stds = {}
        for group in range(NUM_GROUPS):
            
            groupwise_results[group] = np.asarray([self.__results[k]['representation'][self.__delta][group] for k in all_ks])
            if experiment == 'ALG':
                groupwise_stds[group] = np.asarray([0 for k in all_ks])        
            else:
                groupwise_stds[group] = np.asarray([self.__result_stds[k]['representation'][self.__delta][group] for k in all_ks])        


        markerwidthval = 1


        for group in range(NUM_GROUPS):
            # Plot for each group

            k_str = '$k = $'+str(self.__k_for_evaluation[-1])
            
            self.axes_dict[metric1][group].set_title('Group = '+self.__groupname[group]+', '+k_str, fontsize=40)

            # if group == 0:
            #     k_str = '$k = $'+str(self.__k_for_evaluation[-1])
            #     if self.__prot_attr_name == 'sex':
            #         self.axes_dict[metric1][group].set_title('Protected group = female, '+k_str, fontsize=40)
            #     elif self.__prot_attr_name == 'age35':
            #         self.axes_dict[metric1][group].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=40)
            #     elif self.__prot_attr_name == 'age25':
            #         self.axes_dict[metric1][group].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=40)
            #     elif self.__prot_attr_name == 'race':
            #         self.axes_dict[metric1][group].set_title('Protected group = African American, '+k_str, fontsize=40)
            #     else:
            #         self.axes_dict[metric1][group].set_title('Group = '+self.__groupname[group]+', '+k_str, fontsize=40)
            
            self.axes_dict[metric1][group].set_xlabel('top $i$ ranks', fontsize=40)
            self.axes_dict[metric1][group].set_ylabel('representation', fontsize=40)
            self.axes_dict[metric1][group].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            self.axes_dict[metric1][group].tick_params(axis='both', which='major', labelsize=40)
            self.axes_dict[metric1][group].tick_params(axis='both', which='minor', labelsize=40)
            self.axes_dict[metric1][group].set_facecolor("white")
            if self.__dataset == 'jee2009_category':
                if group == 0:
                    self.axes_dict[metric1][group].set_ylim(0.5,0.8)
                else:
                    self.axes_dict[metric1][group].set_ylim(-0.05,0.3)
                
            if experiment == 'ALG':
                lns1 = self.axes_dict[metric1][group].plot(all_ks, 
                                                           groupwise_results[group], 
                                                           color=colormap[experiment], 
                                                           label=labelmap[experiment], 
                                                           marker=markermap[experiment], 
                                                           linestyle=linemap[experiment],
                                                           markersize=markersizemap[experiment], 
                                                           fillstyle=markerfillstyle[experiment], 
                                                           markeredgewidth=markerwidthval)
            else:
                lns1 = self.axes_dict[metric1][group].plot(all_ks, 
                                                           groupwise_results[group], 
                                                           color=colormap[experiment], 
                                                           label=labelmap[experiment], 
                                                           marker=markermap[experiment], 
                                                           linestyle=linemap[experiment],
                                                           markersize=markersizemap[experiment], 
                                                           fillstyle=markerfillstyle[experiment], 
                                                           markeredgewidth=markerwidthval)
                self.axes_dict[metric1][group].fill_between(x=np.array(all_ks), y1=groupwise_results[group]-groupwise_stds[group], y2=groupwise_results[group]+groupwise_stds[group], color=colormap[experiment],alpha=0.35)
        
            self.axes_dict[metric1][group].plot(all_ks, 
                                                   [p[group]]*len(all_ks),  
                                                   label=None, 
                                                   color='red', 
                                                   linestyle='dotted', 
                                                   linewidth=10.0)

            
            # self.axes_dict[metric1][group].legend(prop={'size': 20}, facecolor='white', loc='upper center',ncol=2, framealpha=1)#, bbox_to_anchor=(0.49, 1.1))
            self.figures[metric1][group].savefig(self.__resultDir + metric1+'_' + self.__dataset +'_' + self.__groupname[group]+ '.pdf', bbox_inches = "tight")

        return None


    def __plot_ndcg(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, capsizemap, lns=None):
        
        # Print the values from delta = self.__delta

        all_ks = list(self.__results.keys())


        result1 = np.asarray([self.__results[k]['ndcg'][self.__delta] for k in all_ks])
        result_std1 = np.asarray([self.__result_stds[k]['ndcg'][self.__delta] for k in all_ks])


        
        # if self.__block is not None:
        #     k_str = '$k\' = $'+str(self.__block - self.BLOCK_SIZE+1)+' to '+str(self.__block)
        # else:
        #     k_str = '$k = $'+str(self.__k_for_evaluation[-1])
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
        self.axes_dict[metric1].set_ylabel('ndcg', fontsize=40)
        # self.axes_dict[metric1].yaxis.set_ticks(all_ks)
        self.axes_dict[metric1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=40)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=40)
        self.axes_dict[metric1].set_facecolor("white")
                

        ##### , fillstyle='none',markersize=3
        markerwidthval = 4
        name = experiment
        lns1 = self.axes_dict[metric1].plot(all_ks, result1, color=colormap[experiment], label=labelmap[experiment], marker=markermap[experiment], linestyle=linemap[experiment],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
        if 'ALG' not in experiment:
            self.axes_dict[metric1].fill_between(x=np.array(all_ks), y1=result1-result_std1, y2=result1+result_std1, color=colormap[experiment],alpha=0.35)
        #----- uncomment this if you want legend
        # self.axes_dict[metric1].legend(prop={'size': 50}, facecolor='white', loc='upper center',ncol=4, framealpha=1)#, bbox_to_anchor=(0.49, 1.1))

        # self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.png', dpi=300)
        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.pdf',bbox_inches='tight')

        return lns1

    def __plot_proportions(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, labelmap, lns=None):
        
        # Plots the proportion of groups for stochastic ranks in each rank
        # Values are taken from the delta specified in self.__delta

        all_ks = [val for val in range(1,self.__k_for_evaluation[-1]+1)]
        # all_ks = [val for val in range(1,self.__k_for_evaluation[0]+1)]
        p = self.__true_rep_all 
        groupwise_results = {}
        for group in range(self.__num_groups):
            groupwise_results[group] = [val[group] for val in self.at_k_proportions[self.__delta]]

            # assert len(groupwise_results[group]) == len(all_ks)
        

        for group in range(self.__num_groups):
            # Plot for each group

            k_str = '$k = $'+str(self.__k_for_evaluation[-1])
            
            self.axes_dict[metric1][group].set_title('Group = '+self.__groupname[group]+', '+k_str, fontsize=40)

            # if group == 0:
            #     k_str = '$k = $'+str(self.__k_for_evaluation[-1])

            #     if self.__prot_attr_name == 'sex':
            #         self.axes_dict[metric1][group].set_title('Protected group = female, '+k_str, fontsize=40)
            #     elif self.__prot_attr_name == 'age35':
            #         self.axes_dict[metric1][group].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=40)
            #     elif self.__prot_attr_name == 'age25':
            #         self.axes_dict[metric1][group].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=40)
            #     elif self.__prot_attr_name == 'race':
            #         self.axes_dict[metric1][group].set_title('Protected group = African American, '+k_str, fontsize=40)
            #     else:
            #         self.axes_dict[metric1][group].set_title('Group = '+self.__groupname[group]+', '+k_str, fontsize=40)
        
            self.axes_dict[metric1][group].set_xlabel('at rank $i$', fontsize=40)
            self.axes_dict[metric1][group].set_ylabel('fraction of rankings', fontsize=40)
            self.axes_dict[metric1][group].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            self.axes_dict[metric1][group].tick_params(axis='both', which='major', labelsize=40)
            self.axes_dict[metric1][group].tick_params(axis='both', which='minor', labelsize=40)
            self.axes_dict[metric1][group].set_facecolor("white")
            
            
            markerwidthval = 4

            name = experiment
            lns1 = self.axes_dict[metric1][group].plot(all_ks, groupwise_results[group], color=colormap[experiment], label=labelmap[experiment], linestyle=linemap[experiment])
            self.axes_dict[metric1][group].plot(all_ks, 
                                                   [p[group]]*len(all_ks),  
                                                   label=None, 
                                                   color='red', 
                                                   linestyle='dotted', 
                                                   linewidth=10.0)
            
            # self.axes_dict[metric1][group].legend( prop={'size': 20}, facecolor='white', loc='upper center',ncol=3, framealpha=1)#, bbox_to_anchor=(0.49, 1.1))

            self.figures[metric1][group].savefig(self.__resultDir + metric1+'_' + self.__dataset +'_' + self.__groupname[group]+ '.pdf', bbox_inches = "tight")