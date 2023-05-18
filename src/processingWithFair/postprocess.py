import processingWithFair.rerank_for_fairness as rerank
from processingWithFair.DatasetDescription import DatasetDescription
import sys

class Postprocessing():

    def __init__(self, args):
        self.dataset = args.postprocess[0]
        self.method = args.postprocess[1]
        self.k = args.k
        self.multi_group = args.multi_group
        self.rev_flag = args.rev_flag
        self.deltas = [ 0.1]#0.15, 0.20, 0.25, 0.30]
        self.nsteps = 2
        self.num_samples = args.num_samples
        self.eps = args.eps if hasattr(args, 'eps') else None

    def call_function(self, dataset):

        for i in range(len(self.deltas)):
            sys.stdout.flush()

            if "ALG" in self.method:
                rerank.rerank_alg(
                    dataset, 
                    self.dataset, 
                    p_deviation=self.deltas[i], 
                    iteration=i, 
                    k=self.k,  
                    rev=self.rev_flag)

            elif "FAIR" in self.method:
                rerank.rerank_fair(
                    dataset, 
                    self.dataset, 
                    p_deviation=self.deltas[i], 
                    iteration=i,  
                    rev=self.rev_flag)

            elif "CELIS" in self.method:
                rerank.rerank_celis(
                    dataset, 
                    self.dataset, 
                    p_deviation=self.deltas[i], 
                    iteration=i,  
                    rev=self.rev_flag, 
                    k=self.k)

            elif "EPS_GREEDY" in self.method:
                rerank.rerank_fair_eps_greedy(
                    dataset, 
                    self.dataset, 
                    num_samples=self.num_samples,
                    eps=self.eps,
                    p_deviation=self.deltas[i], 
                    rev=self.rev_flag, 
                    k=self.k)

            elif "LATTICE" in self.method:
                rerank.rerank_lattice(
                    dataset, 
                    self.dataset, 
                    num_samples=self.num_samples,
                    p_deviation=self.deltas[i], 
                    rev=self.rev_flag, 
                    k=self.k)
            
            elif "PREFIX" in self.method:
                rerank.rerank_prefix_lattice(
                    dataset, 
                    self.dataset, 
                    num_samples=self.num_samples,
                    p_deviation=self.deltas[i], 
                    nsteps = self.nsteps,
                    rev=self.rev_flag, 
                    k=self.k)

            elif "DP" in self.method:
                rerank.rerank_DP(
                    dataset, 
                    self.dataset, 
                    num_samples=self.num_samples,
                    p_deviation=self.deltas[i], 
                    rev=self.rev_flag, 
                    k=self.k)

            elif "RANDOMIZED" in self.method:
                rerank.rerank_randomized(
                    dataset, 
                    self.dataset, 
                    num_samples=self.num_samples,
                    p_deviation=self.deltas[i], 
                    rev=self.rev_flag, 
                    k=self.k)





    def postprocess(self):

        if self.dataset == "german" and not self.multi_group:

            """
            German Credit dataset - age 25
            """

            print("Start reranking of German Credit - Age 25")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age25"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age25']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age25.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age25_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age25_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            self.call_function(dataset=GermanCreditData)

            """
            German Credit dataset - age 35
            """
            print("Start reranking of German Credit - Age 35")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age35"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age35']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age35.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age35_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age35_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            self.call_function(dataset=GermanCreditData)

        elif self.dataset == "german" and self.multi_group:
            """
            German Credit dataset - age
            """
            print("Start reranking of German Credit - Age")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            self.call_function(dataset=GermanCreditData)

            """
            German Credit dataset - age_gender
            """
            print("Start reranking of German Credit - Age and Gender")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age_gender"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age_gender']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age_gender.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age_gender_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age_gender_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            self.call_function(dataset=GermanCreditData)

        elif self.dataset == "jee2009" and not self.multi_group:

            """
            Jee2009 dataset - gender
            """

            print("Start reranking of JEE2009 - Gender")
            protected_attribute = 1
            score_attribute = 2
            protected_group = "gender"
            header = ['id', 'gender', 'mark']
            judgment = "mark"

            origFile = "../data/JEE2009/JEE2009_gender.csv"
            if self.rev_flag:
                resultFile = "../data/JEE2009/JEE2009_gender_rev_" + self.method 
            else:
                resultFile = "../data/JEE2009/JEE2009_gender_" + self.method 
            JEE2009Data = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            self.call_function(dataset=JEE2009Data)

            
        elif self.dataset == "jee2009" and self.multi_group:
            """
            Jee2009 dataset - category
            """
            print("Start reranking of JEE2009 - Category")
            protected_attribute = 1
            score_attribute = 2
            protected_group = "category"
            header = ['id', 'category', 'mark']
            judgment = "mark"

            origFile = "../data/JEE2009/JEE2009_category.csv"
            if self.rev_flag:
                resultFile = "../data/JEE2009/JEE2009_category_rev_" + self.method 
            else:
                resultFile = "../data/JEE2009/JEE2009_category_" + self.method 
            JEE2009Data = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            self.call_function(dataset=JEE2009Data)


            # """
            # Jee2009 dataset - category_gender
            # """
            # print("Start reranking of JEE2009 - Category-Gender")
            # protected_attribute = 1
            # score_attribute = 2
            # protected_group = "category_gender"
            # header = ['id', 'category_gender', 'mark']
            # judgment = "mark"

            # origFile = "../data/JEE2009/JEE2009_category_gender.csv"
            # if self.rev_flag:
            #     resultFile = "../data/JEE2009/JEE2009_category_gender_rev_" + self.method 
            # else:
            #     resultFile = "../data/JEE2009/JEE2009_category_gender_" + self.method 
            # JEE2009Data = DatasetDescription(resultFile,
            #                                          origFile,
            #                                          protected_attribute,
            #                                          score_attribute,
            #                                          protected_group,
            #                                          header,
            #                                          judgment)
            
            # self.call_function(dataset=JEE2009Data)
        
        elif self.dataset == 'compas':

            """
            COMPAS propublica dataset - race
            """
            print("Start reranking of COMPAS propublica - Race")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "race"
            header = ['priors_count','Violence_rawscore','Recidivism_rawscore','race']
            judgment = "Recidivism_rawscore"

            origFile = "../data/COMPAS/ProPublica_race.csv"
            if self.rev_flag:
                resultFile = "../data/COMPAS/ProPublica_race_rev_" + self.method 
            else:
                resultFile = "../data/COMPAS/ProPublica_race_" + self.method 
            CompasData = DatasetDescription(resultFile,
                                             origFile,
                                             protected_attribute,
                                             score_attribute,
                                             protected_group,
                                             header,
                                             judgment)
            
            self.call_function(dataset=CompasData)

            """
            COMPAS propublica dataset - gender
            """
            print("Start reranking of COMPAS propublica - gender")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "sex"
            header = ['priors_count','Violence_rawscore','Recidivism_rawscore','sex']
            judgment = "Recidivism_rawscore"

            origFile = "../data/COMPAS/ProPublica_sex.csv"
            if self.rev_flag:
                resultFile = "../data/COMPAS/ProPublica_sex_rev_" + self.method 
            else:
                resultFile = "../data/COMPAS/ProPublica_sex_" + self.method 
            CompasData = DatasetDescription(resultFile,
                                             origFile,
                                             protected_attribute,
                                             score_attribute,
                                             protected_group,
                                             header,
                                             judgment)
            
            self.call_function(dataset=CompasData)