from sklearn.preprocessing import LabelBinarizer

import argparse as ap
import config 
import pandas as pd


class InputDataPreprocessor:
    def __init__(self, transformed_policy_holders_df, input_csv_path=None, output_csv_path=None):
        if input_csv_path is None:
            self.input_csv_path = config.CHATBOT_INPUT_CSV_FILEPATH
        if output_csv_path is None:
            self.output_csv_path = config.CHATBOT_OUTPUT_CSV_FILEPATH
        self.user_input_df = pd.read_csv(self.input_csv_path, dtype=object)
        self.transformed_policy_holders_df = transformed_policy_holders_df
        self.transformed_user_input_df = None
        self.preprocess()
        
    def load_data(self):
        return self.user_input_df
    
    def preprocess(self):
        self.transformed_user_input_df = \
            self.user_input_df[config.POLICY_NUMBER_COL].apply(
                lambda policy_number: self.transformed_policy_holders_df.loc[policy_number])
        self.transformed_user_input_df = self.transformed_user_input_df[config.FINAL_FEATURES]
        
    def load_transformed_data(self):
        return self.transformed_user_input_df
    
    def save_transformed_data(self):
        self.transformed_user_input_df.to_csv(config.CHATBOT_OUTPUT_CSV_FILEPATH, index=False)

    
class PreprocessorArgumentParser:
    def __init__(self):
        self.parser = ap.ArgumentParser()
        self.parser.add_argument('--input_csv_path', dest='input_csv_path', type=str)
        self.parser.add_argument('--output_csv_path', dest='output_csv_path', type=str)
        
    def parse_args(self):
        return self.parser.parse_args()

    
class PolicyHoldersDataRecords:
    def __init__(self):
        self.df = pd.read_csv(config.POLICY_HOLDERS_DATA_CSV_PATH, dtype=object)
        self.transformed_df = None
        self.preprocess()
              
    def preprocess(self):
        for col in [config.MONTH_COL, config.MONTH_CLAIMED_COL]:
            self.df[col] = pd.to_datetime(
                self.df[col], format="%b", errors="coerce").dt.strftime("%m") + \
                self.df[col]
     
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
        
        self.transformed_df.index = self.df[config.POLICY_NUMBER_COL]
        
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
    
    print("Started running preprocessing of user input data.")
    args = PreprocessorArgumentParser().parse_args()
    
    policy_holders_data_records = PolicyHoldersDataRecords()
    policy_holders_df = policy_holders_data_records.load_data()
    transformed_policy_holders_df = policy_holders_data_records.load_transformed_data()
    
    input_data_preprocessor = InputDataPreprocessor(
        input_csv_path=args.input_csv_path, 
        output_csv_path=args.output_csv_path, 
        transformed_policy_holders_df=transformed_policy_holders_df)
    user_input_df = input_data_preprocessor.load_data()
    transformed_user_input_df = input_data_preprocessor.load_transformed_data()
    input_data_preprocessor.save_transformed_data()
    
    print("Done running preprocessing of user input data.")
    print("Preprocessed data is stored in {}".format(config.CHATBOT_OUTPUT_CSV_FILEPATH))
    