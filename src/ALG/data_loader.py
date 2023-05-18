import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


GERMAN_FILE = "../data/GermanCredit/german_credit_data.csv"

PROTECT = "gender" # Choose between gender or highschool
ENGINEERING_FILE = f"../data/EngineeringStudents/NoSemiPrivate/{PROTECT}/fold_1/"


class Data:
    def __init__(self, args):

        self.data_name = args.data.lower()
        self.k = args.k
        self.alpha = args.alpha
        self.shuffle = args.shuffle
        self.num_groups = None
        self.EPSILON = None
        self.final_rank = None
    
    def print_params(self):
        print("k:\t",self.k)
        print("alpha:\t",self.alpha)
        print("num_groups:\t",self.num_groups)
        print("epsilon:\t",self.EPSILON)
        print("shuffle:\t",self.shuffle)

    def load_data(self, q, eps_flag=False):

        if "german" in self.data_name:

            # Load the dataset file
            df = pd.read_csv(GERMAN_FILE, index_col=0)
            df = df.fillna(value="NA")
            self.num_groups = len(df.groupby(self.args.group).count())
            self.alpha = max(self.alpha, 1.0/self.num_groups + 0.05)
            self.k = max(self.k, 2*(1+1.0/(self.alpha - 1.0/self.num_groups)))
            self.EPSILON = (2.0/self.k)*(1+1.0/(self.alpha - 1.0/self.num_groups))
            if eps_flag == True:
                self.EPSILON += q

            # convert categorical to one-hot and scale other columns
            preprocess = make_column_transformer(
                (StandardScaler(), ['Age', 'Credit amount', 'Duration']),
                (OneHotEncoder(sparse=False), ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']),
                remainder="passthrough"
            )
            mat = preprocess.fit_transform(df)

            if self.shuffle:
                # Shuffle the original data before sorting
                shuffle_id = [idx for idx in range(mat.shape[0])]
                np.random.shuffle(shuffle_id)
                mat = mat[shuffle_id]


            # Sort the dataset based on relevance to assign global ranking
            sorted_ids = np.argsort(-mat[:, -1])
            mat = mat[sorted_ids]

            # Extract the features, relevance
            PROTECTED_ID = 4
            self.features = mat[:, :-2] # (Age -- Purpose)
            self.protected_feature = mat[:, PROTECTED_ID] # (Sex: 1 - Male, 0 - Female )
            self.relevance = mat[:, -1] # (Risk: 1 - Good, 0 - Bad)
            self.data_id = [f"id-{idx+1}" for idx in range(len(self.relevance))]

            self.protected_mapping = {id: gender for id, gender in zip(self.data_id, self.protected_feature)}
            self.relevance_mapping = {id: rel for id, rel in zip(self.data_id, self.relevance)}

        elif "engineering" in self.data_name:

            train_file = ENGINEERING_FILE + f"chileDataL2R_{PROTECT}_nosemi_fold1_train.txt"
            test_file = ENGINEERING_FILE + f"chileDataL2R_{PROTECT}_nosemi_fold1_test.txt"

            # Setting query id 1 manually here.


            mat = np.loadtxt(test_file, delimiter=',')
            self.num_groups = 2
            self.alpha = max(self.alpha, 1.0/self.num_groups + 0.05)
            self.k = max(self.k, 2*int(1+1.0/(self.alpha - 1.0/self.num_groups)+1))
            self.EPSILON = (2.0/self.k)*(1+1.0/(self.alpha - 1.0/self.num_groups))
            if eps_flag == True:
                self.EPSILON += q
            

            # Sort the dataset based on relevance to assign global ranking
            sorted_ids = np.argsort(-mat[:, -1])
            mat = mat[sorted_ids]

            # Extract the features, relevance
            PROTECTED_ID = 1
            self.features = mat[:, 1:-2] # (PSU scores)
            self.protected_feature = mat[:, PROTECTED_ID]
            self.relevance = mat[:, -1] 
            self.data_id = [f"id-{idx+1}" for idx in range(len(self.relevance))]

            self.protected_mapping = {id: gender for id, gender in zip(self.data_id, self.protected_feature)}
            self.relevance_mapping = {id: rel for id, rel in zip(self.data_id, self.relevance)}

        else:
            raise ValueError("Dataset is not correctly specified")