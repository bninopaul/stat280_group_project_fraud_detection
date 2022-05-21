import argparse as ap
import config 
import pandas as pd

class InputDataPreprocessor:
    def __init__(self, input_csv_path):
        
        

class PreprocessorArgumentParser:
    def __init__(self):
        self.parser = ap.ArgumentParser()
        self.parser.add_argument('--input_csv_path', dest='input_csv_path', type=str)
        
    def parse_args(self):
        return self.parser.parse_args()

class PolicyHoldersDataRecords:
    def __init__(self):
        self.df = pd.read_csv(config.POLICY_HOLDERS_DATA_CSV_PATH, dtype=object)
        self.transformed_df = None
        self.preprocess()
              
    def preprocess(self):
        self.df["Month"] = pd.to_datetime(
            self.df["Month"], format="%b", errors="coerce").dt.strftime("%m") + \
            self.df["Month"]
        self.df["MonthClaimed"] = pd.to_datetime(
            self.df["MonthClaimed"], format="%b", errors="coerce").dt.strftime("%m") + \
            self.df["MonthClaimed"]
        for col in config.NUMERIC_COLS:
            self.df[col] = pd.to_numeric(self.df[col])    
            
        # drop the rows with null feature values
        self.df = self.df.dropna().reset_index(drop=True)
        
        # binarize or one-hot-encode the categorical features
        cat_label_binarizer = CategoricalLabelBinarizer(config.CAT_FEAT_2_LAB_BIN_PATH)
        categorical_features = config.CATEGORICAL_FEATURES
        if cat_label_binarizer.cat_feat_2_lab_bin is None:
            cat_label_binarizer.fit(self.df, categorical_features)
        self.transformed_df = cat_label_binarizer.transform(self.df, categorical_features)
        
        # append the numerical features and the target variable
        numerical_features = config.NUMERICAL_FEATURES
        fraud_col = config.FRAUD_COL
        self.transformed_df = pd.concat(
            [self.transformed_df, 
             self.df[numerical_features], 
             self.df[fraud_col]], axis=1)
        
    def load_data(self):
        return self.df 
    
    def load_transformed_data(self):
        return self.transformed_df
    
class CategoricalLabelBinarizer:
    
    def __init__(self, cat_feat_2_lab_bin_path=None):
        cat_feat_2_lab_bin = pd.read_pickle(cat_feat_2_lab_bin_path)
        self.cat_feat_2_lab_bin = cat_feat_2_lab_bin
    
    def fit(self, df, categorical_features):
        for feat_col in categorical_features:
            lb = LabelBinarizer()
            lb.fit(df[feat_col])        
            self.cat_feat_2_lab_bin[feat_col] = lb
    
    def transform(self, df, categorical_features):
        transformed_df = pd.DataFrame([])
        for feat_col in categorical_features:
            lb = self.cat_feat_2_lab_bin[feat_col]
            if len(lb.classes_) > 2:
                columns = [feat_col + "_" + str(col_val) 
                           for col_val in lb.classes_]
            else:
                columns = [feat_col + "_" + str(col_val) 
                           for col_val in lb.classes_[:-1]]
            feat_df = pd.DataFrame(
                lb.transform(df[feat_col]), columns=columns)
            transformed_df = pd.concat([transformed_df, feat_df], axis=1)
        return transformed_df
        
if __name__ == '__main__':

    args = PreprocessorArgumentParser().parse_args()
    df = PolicyHoldersDataRecords().load_data()
    transformed_df = PolicyHoldersDataRecords().load_transformed_data()
    
    
    
    print(df.dtypes)
    
    

